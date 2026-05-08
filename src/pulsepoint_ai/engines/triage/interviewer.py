"""Symptom interviewer: LLM-first with LoRA + rule-based fallbacks.

Primary path: Gemini via LLMClient using configs/prompts/interviewer_v1.j2.
Fallback 1: fine-tuned FLAN-T5-small LoRA adapter (offline, lower quality).
Fallback 2: deterministic rules covering common chief complaints.
"""
from __future__ import annotations

import os
import threading
import uuid
from typing import Any

from pulsepoint_ai.core.config import get_models_config, get_settings
from pulsepoint_ai.core.schemas.common import PatientProfile
from pulsepoint_ai.core.schemas.interview import (
    AnswerType,
    AnsweredQuestion,
    InterviewerRequest,
    InterviewerResponse,
)
from pulsepoint_ai.llm.prompts import render_prompt


class SymptomInterviewer:
    def __init__(self, device: str = "cpu") -> None:
        self.cfg = get_models_config()["interviewer"]
        self.settings = get_settings()
        self.device = device
        self._model = None
        self._tokenizer = None
        self._load_attempted = False
        self._load_failed_reason: str | None = None
        self._lock = threading.Lock()

    def _load_model(self):
        """Loads the FLAN-T5 model with LoRA adapters. Idempotent + thread-safe."""
        if self._model is not None or self._load_attempted:
            return

        with self._lock:
            if self._model is not None or self._load_attempted:
                return
            self._load_attempted = True

            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                from peft import PeftModel
            except ImportError as e:
                self._load_failed_reason = f"missing inference deps: {e}"
                print(f"Interviewer LoRA skipped — {self._load_failed_reason}.")
                return

            base_model_path = self.cfg["base_model"]
            adapter_path = self.cfg["adapter"]

            if not os.path.isdir(adapter_path) or not os.listdir(adapter_path):
                self._load_failed_reason = f"adapter directory missing or empty at {adapter_path}"
                print(f"Interviewer LoRA skipped — {self._load_failed_reason}.")
                return

            try:
                print(f"Loading Symptom Interviewer LoRA (base: {base_model_path}, adapter: {adapter_path})...")
                self._tokenizer = AutoTokenizer.from_pretrained(adapter_path)
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float32,
                )
                self._model = PeftModel.from_pretrained(base_model, adapter_path)
                self._model.to(self.device)
                self._model.eval()
                print("Symptom Interviewer LoRA ready (fallback only).")
            except Exception as e:
                self._model = None
                self._tokenizer = None
                self._load_failed_reason = f"load error: {e}"
                print(f"Interviewer LoRA load failed — {self._load_failed_reason}.")

    async def generate_question_llm(
        self,
        symptoms: list[str],
        profile: PatientProfile,
        answered: list[AnsweredQuestion],
        relay_mode: bool,
        llm: Any,
    ) -> dict[str, Any] | None:
        """Primary path: returns {question, rationale, expected_answer_type} or None on failure."""
        if llm is None:
            return None
        try:
            prompt = render_prompt(
                "interviewer_v1",
                symptoms=symptoms,
                patient_profile=profile.model_dump(),
                answered=[a.model_dump() for a in answered],
                relay_mode=relay_mode,
            )
            data = await llm.complete_json(prompt, prompt_version="interviewer_v1")

            question = (data.get("question") or "").strip()
            rationale = (data.get("rationale") or "").strip()
            answer_type_raw = (data.get("expected_answer_type") or "free_text").strip().lower()

            if not question or len(question) < 5:
                return None

            try:
                answer_type = AnswerType(answer_type_raw)
            except ValueError:
                answer_type = AnswerType.FREE_TEXT

            asked_questions = {a.question.strip().lower() for a in answered}
            if question.lower() in asked_questions:
                return None

            return {
                "question": question,
                "rationale": rationale or "Most diagnostically informative next question.",
                "expected_answer_type": answer_type,
            }
        except Exception as e:
            print(f"Interviewer LLM path failed: {e}. Falling back to LoRA/rules.")
            return None

    def generate_question_lora(
        self,
        symptoms: list[str],
        profile: PatientProfile,
        answered: list[AnsweredQuestion],
    ) -> str | None:
        """LoRA fallback. Returns question text or None."""
        self._load_model()
        if self._model is None or self._tokenizer is None:
            return None

        try:
            import torch

            prompt = self._build_lora_prompt(symptoms, profile, answered)
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=int(self.cfg.get("max_new_tokens", 64)),
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3,
                )

            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if not response or len(response) < 5:
                return None

            asked = {a.question.strip().lower() for a in answered}
            if response.lower() in asked:
                return None
            return response
        except Exception as e:
            print(f"Interviewer LoRA inference failed: {e}.")
            return None

    def _build_lora_prompt(
        self,
        symptoms: list[str],
        profile: PatientProfile,
        answered: list[AnsweredQuestion],
    ) -> str:
        """Matches the format used during LoRA training (see scripts/train_interviewer.py)."""
        symptom_list = ", ".join(symptoms)
        instruction = (
            "Act as a physician conducting a clinical interview. Based on the provided "
            "symptoms, ask the single most diagnostically useful follow-up question."
        )
        return f"Instruction: {instruction}\nInput: {symptom_list}\nResponse:"

    def generate_question_rules(
        self,
        symptoms: list[str],
        profile: PatientProfile,
        answered: list[AnsweredQuestion],
    ) -> tuple[str, str, AnswerType]:
        """Last-resort deterministic rules. Returns (question, rationale, answer_type)."""
        if not symptoms:
            return (
                "Could you describe your main symptom in your own words?",
                "Need a chief complaint to start triage.",
                AnswerType.FREE_TEXT,
            )

        joined = " ".join(s.lower() for s in symptoms)
        asked = {a.question.strip().lower() for a in answered}

        candidates: list[tuple[str, str, AnswerType]] = []

        if "chest" in joined:
            candidates += [
                ("Does the pain radiate to your left arm, jaw, or back?",
                 "Radiation pattern discriminates ACS from musculoskeletal pain.", AnswerType.YES_NO),
                ("On a scale of 0 to 10, how severe is the pain right now?",
                 "Severity helps risk-stratify cardiac vs benign causes.", AnswerType.SCALE),
                ("Does the pain get worse when you take a deep breath?",
                 "Pleuritic pain points toward PE or pericarditis.", AnswerType.YES_NO),
            ]
        if "abdominal" in joined or "stomach" in joined or "belly" in joined:
            candidates += [
                ("Did the pain start near your navel and move to the right lower side?",
                 "Migration to RLQ is classic for appendicitis.", AnswerType.YES_NO),
                ("Have you vomited or lost your appetite since the pain started?",
                 "Anorexia/vomiting raises suspicion for surgical abdomen.", AnswerType.YES_NO),
                ("On a scale of 0 to 10, how severe is the pain at its worst?",
                 "Severity drives urgency tier.", AnswerType.SCALE),
            ]
        if "headache" in joined or "head pain" in joined:
            candidates += [
                ("Is this the worst headache of your life, or did it start very suddenly?",
                 "Thunderclap headache is a red flag for SAH.", AnswerType.YES_NO),
                ("Do you have any new vision changes, weakness, or numbness?",
                 "Focal neuro signs flag stroke or mass effect.", AnswerType.YES_NO),
                ("Is there any fever or stiffness in your neck?",
                 "Meningismus suggests meningitis.", AnswerType.YES_NO),
            ]
        if "shortness" in joined or "breath" in joined or "dyspnea" in joined:
            candidates += [
                ("Does the breathlessness get worse when you lie flat?",
                 "Orthopnea points to heart failure.", AnswerType.YES_NO),
                ("Is one of your legs swollen, red, or painful?",
                 "Unilateral leg swelling raises PE suspicion.", AnswerType.YES_NO),
                ("On a scale of 0 to 10, how hard is it to breathe right now?",
                 "Severity drives ED routing.", AnswerType.SCALE),
            ]
        if "fever" in joined or "chills" in joined:
            candidates += [
                ("How many days have you had the fever?",
                 "Duration shifts differential between viral and bacterial.", AnswerType.FREE_TEXT),
                ("What was the highest temperature you measured?",
                 "Magnitude flags bacteremia risk.", AnswerType.FREE_TEXT),
                ("Do you have any new cough, burning urination, or skin rash?",
                 "Localizing symptoms point to source of infection.", AnswerType.YES_NO),
            ]
        if "cough" in joined:
            candidates += [
                ("Are you coughing up blood, yellow/green sputum, or is it dry?",
                 "Sputum character distinguishes pneumonia, bronchitis, TB.", AnswerType.FREE_TEXT),
                ("How many days or weeks have you had the cough?",
                 "Duration separates acute from chronic causes.", AnswerType.FREE_TEXT),
                ("Have you had any fever, weight loss, or night sweats?",
                 "B-symptoms flag TB, malignancy.", AnswerType.YES_NO),
            ]
        if "dizz" in joined or "vertigo" in joined or "spinning" in joined:
            candidates += [
                ("Does the room feel like it is spinning, or do you feel faint?",
                 "Vertigo vs presyncope splits the differential cleanly.", AnswerType.FREE_TEXT),
                ("Does the dizziness get worse when you change head position?",
                 "Positional trigger suggests BPPV.", AnswerType.YES_NO),
                ("Have you had any new weakness, numbness, or trouble speaking?",
                 "Focal signs flag posterior-circulation stroke.", AnswerType.YES_NO),
            ]
        if "back" in joined:
            candidates += [
                ("Do you have any numbness, weakness, or loss of bladder/bowel control?",
                 "Cauda equina red flags require emergent imaging.", AnswerType.YES_NO),
                ("Did the pain start after a fall, lift, or twist?",
                 "Mechanism distinguishes mechanical from atraumatic causes.", AnswerType.YES_NO),
                ("Does the pain shoot down one of your legs past the knee?",
                 "Radicular pattern suggests disc herniation.", AnswerType.YES_NO),
            ]
        if "rash" in joined or "skin" in joined or "itch" in joined:
            candidates += [
                ("Is the rash spreading, and did it start in one specific area?",
                 "Spread pattern separates contact dermatitis from systemic causes.", AnswerType.FREE_TEXT),
                ("Have you started any new medications or foods in the last 2 weeks?",
                 "Allergic exposure is the highest-yield history.", AnswerType.YES_NO),
                ("Is the rash painful, blistering, or accompanied by fever?",
                 "Pain/blisters/fever raise SJS/cellulitis suspicion.", AnswerType.YES_NO),
            ]

        for q, r, t in candidates:
            if q.lower() not in asked:
                return q, r, t

        main = symptoms[0].lower()
        return (
            f"When did the {main} start, and has it been constant or come and go?",
            "Onset and pattern are the highest-yield history when no template fits.",
            AnswerType.FREE_TEXT,
        )


# Global instance for easy access (load lazily on first request)
interviewer = SymptomInterviewer()


async def next_question(req: InterviewerRequest, llm: Any = None) -> InterviewerResponse:
    """Generates the next clinical question and formats it as an InterviewerResponse.

    Tries Gemini (primary) -> LoRA adapter -> deterministic rules.
    """
    llm_result = await interviewer.generate_question_llm(
        req.symptoms, req.patient_profile, req.answered, req.relay_mode, llm
    )
    if llm_result is not None:
        primary_model = interviewer.cfg.get("llm_label", "gemini-flash-latest")
        return InterviewerResponse(
            request_id=str(uuid.uuid4()),
            question=llm_result["question"],
            rationale=llm_result["rationale"],
            expected_answer_type=llm_result["expected_answer_type"],
            model_version=f"llm-{primary_model}",
        )

    lora_q = interviewer.generate_question_lora(
        req.symptoms, req.patient_profile, req.answered
    )
    if lora_q is not None:
        return InterviewerResponse(
            request_id=str(uuid.uuid4()),
            question=lora_q,
            rationale="Generated by fine-tuned FLAN-T5-small LoRA (LLM unavailable).",
            expected_answer_type=AnswerType.FREE_TEXT,
            model_version=f"flan-t5-lora-{interviewer.cfg.get('version', 'v1.0.0')}",
        )

    rule_q, rule_r, rule_t = interviewer.generate_question_rules(
        req.symptoms, req.patient_profile, req.answered
    )
    return InterviewerResponse(
        request_id=str(uuid.uuid4()),
        question=rule_q,
        rationale=rule_r,
        expected_answer_type=rule_t,
        model_version="rules_v1",
    )
