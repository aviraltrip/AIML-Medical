"""Inference for the fine-tuned FLAN-T5 Small LoRA clinical interviewer.

This is the showcase model — generates the most diagnostically relevant
follow-up question. Falls back to a rule-based question if the adapter
files or torch/transformers/peft are unavailable.
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
                print(f"Interviewer load skipped — {self._load_failed_reason}. Using fallback.")
                return

            base_model_path = self.cfg["base_model"]
            adapter_path = self.cfg["adapter"]

            if not os.path.isdir(adapter_path) or not os.listdir(adapter_path):
                self._load_failed_reason = f"adapter directory missing or empty at {adapter_path}"
                print(f"Interviewer load skipped — {self._load_failed_reason}. Using fallback.")
                return

            try:
                print(f"Loading Symptom Interviewer (base: {base_model_path}, adapter: {adapter_path})...")
                self._tokenizer = AutoTokenizer.from_pretrained(adapter_path)
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float32,
                )
                self._model = PeftModel.from_pretrained(base_model, adapter_path)
                self._model.to(self.device)
                self._model.eval()
                print("Symptom Interviewer ready.")
            except Exception as e:
                self._model = None
                self._tokenizer = None
                self._load_failed_reason = f"load error: {e}"
                print(f"Interviewer load failed — {self._load_failed_reason}. Using fallback.")

    def generate_question(
        self,
        symptoms: list[str],
        profile: PatientProfile,
        answered: list[AnsweredQuestion] | None = None,
    ) -> tuple[str, str]:
        """Returns (question, source_tag) where source_tag indicates which path generated it."""
        self._load_model()

        if self._model is None or self._tokenizer is None:
            return self._generate_fallback(symptoms, profile, answered or []), "fallback_rules"

        try:
            import torch

            prompt = self._build_prompt(symptoms, profile, answered or [])
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
                    do_sample=True,
                    temperature=0.3,
                    repetition_penalty=2.5,
                    no_repeat_ngram_size=3,
                    top_p=0.9,
                )

            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if not response or len(response) < 5:
                return self._generate_fallback(symptoms, profile, answered or []), "fallback_rules"
            return response, "lora_adapter"
        except Exception as e:
            print(f"Interviewer inference failed: {e}. Falling back.")
            return self._generate_fallback(symptoms, profile, answered or []), "fallback_rules"

    def _build_prompt(
        self,
        symptoms: list[str],
        profile: PatientProfile,
        answered: list[AnsweredQuestion],
    ) -> str:
        """Matches the exact format used during LoRA training (see scripts/train_interviewer.py)."""
        symptom_list = ", ".join(symptoms)
        instruction = (
            "Act as a physician conducting a clinical interview. Based on the provided "
            "symptoms, ask the single most diagnostically useful follow-up question."
        )
        return f"Instruction: {instruction}\nInput: {symptom_list}\nResponse:"

    def _generate_fallback(
        self,
        symptoms: list[str],
        profile: PatientProfile,
        answered: list[AnsweredQuestion],
    ) -> str:
        if not symptoms:
            return "Could you describe your main symptom in your own words?"
        main = symptoms[0].lower()
        if "chest" in main:
            return "Does the pain radiate to your left arm or jaw?"
        if "abdominal" in main or "stomach" in main:
            return "When did the pain start, and is it constant or intermittent?"
        if "headache" in main:
            return "Is the headache throbbing, stabbing, or a dull ache?"
        if "shortness" in main or "breath" in main:
            return "Does the breathlessness get worse when you lie flat or with exertion?"
        if "fever" in main:
            return "How long have you had the fever, and have you measured a temperature?"
        return f"Could you tell me more about the onset of your {main}?"


# Global instance for easy access (load lazily on first request)
interviewer = SymptomInterviewer()


async def next_question(req: InterviewerRequest, llm: Any = None) -> InterviewerResponse:
    """Generates the next clinical question and formats it as an InterviewerResponse."""
    question_text, source = interviewer.generate_question(
        req.symptoms, req.patient_profile, req.answered
    )

    rationale = (
        "Generated by fine-tuned FLAN-T5-small LoRA clinical interviewer."
        if source == "lora_adapter"
        else "Rule-based fallback (LoRA adapter unavailable in this environment)."
    )
    version = (
        f"flan-t5-lora-{interviewer.cfg.get('version', 'v1.0.0')}"
        if source == "lora_adapter"
        else "fallback_v1"
    )

    return InterviewerResponse(
        request_id=str(uuid.uuid4()),
        question=question_text,
        rationale=rationale,
        expected_answer_type=AnswerType.FREE_TEXT,
        model_version=version,
    )
