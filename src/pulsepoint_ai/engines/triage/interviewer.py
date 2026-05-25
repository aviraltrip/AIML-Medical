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
        asked = {a.question.strip().lower() for a in answered}

        candidates: list[tuple[str, str, AnswerType]] = [
            (
                "Is there a history of sugar (diabetes) or high BP in your parents or siblings?",
                "Assesses genetic predisposition for chronic metabolic conditions.",
                AnswerType.YES_NO
            ),
            (
                "How much physical work, farming, walking, or exercise do you do in a day?",
                "Assesses physical activity level to calculate IDRS score.",
                AnswerType.FREE_TEXT
            ),
            (
                "Do you regularly chew tobacco, smoke bidi, or drink alcohol?",
                "Assesses behavioral risk factors for vascular health and metabolic disease.",
                AnswerType.YES_NO
            ),
            (
                "When was your blood pressure or blood sugar last checked, and what was the value?",
                "Retrieves historical clinical baseline values if available.",
                AnswerType.FREE_TEXT
            ),
            (
                "How far is your village or home from the nearest Primary Health Centre (PHC)?",
                "Screens for healthcare accessibility and referral delay risk.",
                AnswerType.FREE_TEXT
            ),
            (
                "Have you noticed having frequent urination at night, dry mouth, or excessive thirst?",
                "Screens for hallmark symptoms of hyperglycemia.",
                AnswerType.YES_NO
            ),
            (
                "Do you regularly get headaches, dizziness, or chest tightness when working?",
                "Screens for common somatic indicators of high blood pressure.",
                AnswerType.YES_NO
            ),
            (
                "Do you have any slow-healing sores or ulcers on your feet?",
                "Screens for diabetic peripheral neuropathy or microvascular complications.",
                AnswerType.YES_NO
            ),
            (
                "What is your typical diet? Do you consume high-salt, fried, or wheat/rice-heavy meals?",
                "Evaluates nutritional risks for metabolic syndrome.",
                AnswerType.FREE_TEXT
            ),
            (
                "Have you recently experienced sudden blurriness or difficulty reading, even with glasses?",
                "Screens for early diabetic retinopathy or vascular changes.",
                AnswerType.YES_NO
            ),
            (
                "Do you feel any burning, tingling, or loss of sensation in your hands or feet?",
                "Screens for diabetic peripheral neuropathy symptoms.",
                AnswerType.YES_NO
            ),
            (
                "If you have been pregnant, did you ever have high blood sugar or a baby weighing over 4 kg?",
                "Assesses risk of gestational diabetes history which increases Type 2 Diabetes risk.",
                AnswerType.YES_NO
            ),
            (
                "Have you noticed dark, velvety patches of skin around your neck or armpits?",
                "Screens for acanthosis nigricans, a hallmark sign of insulin resistance.",
                AnswerType.YES_NO
            ),
            (
                "Do you snore loudly at night or feel extremely sleepy during the daytime?",
                "Identifies potential sleep apnea, highly correlated with metabolic syndrome.",
                AnswerType.YES_NO
            ),
            (
                "What is your primary mode of travel to the PHC, and how much does it cost you?",
                "Socioeconomic accessibility screening for default risk.",
                AnswerType.FREE_TEXT
            ),
            (
                "What is your approximate weight and height, or do you consider yourself overweight?",
                "Assesses BMI indicators for diabetic staging.",
                AnswerType.FREE_TEXT
            )
        ]

        for q, r, t in candidates:
            if q.lower() not in asked:
                return q, r, t

        return (
            "Could you tell me if you have any other concerns about sugar or blood pressure?",
            "Fallback question to gather general chronic health concerns.",
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
