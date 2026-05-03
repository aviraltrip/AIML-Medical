"""Inference for the fine-tuned FLAN-T5 Small LoRA clinical interviewer.

This model is the "Showcase" of the hackathon—it generates the most 
diagnically relevant follow-up question.
"""
from __future__ import annotations

import os
import uuid
from typing import Any
from pulsepoint_ai.core.config import get_models_config, get_settings
from pulsepoint_ai.core.schemas.common import PatientProfile
from pulsepoint_ai.core.schemas.interview import InterviewerRequest, InterviewerResponse, AnswerType

class SymptomInterviewer:
    def __init__(self, device: str = "cpu") -> None:
        self.cfg = get_models_config()["interviewer"]
        self.settings = get_settings()
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Loads the FLAN-T5 model with LoRA adapters."""
        if self._model:
            return
            
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from peft import PeftModel

        base_model_path = self.cfg["base_model"]
        adapter_path = self.cfg["adapter"]

        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"LoRA adapter not found at {adapter_path}. Please run the training script first.")

        print(f"Loading Symptom Interviewer (Base: {base_model_path})...")
        self._tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # Load base model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        )
        
        # Load LoRA adapter
        self._model = PeftModel.from_pretrained(model, adapter_path)
        self._model.to(self.device)
        self._model.eval()

    def generate_question(self, symptoms: list[str], profile: PatientProfile) -> str:
        """Generates a targeted clinical follow-up question."""
        try:
            self._load_model()
        except Exception as e:
            print(f"Model loading failed: {e}. Falling back to pre-defined logic.")
            return self._generate_fallback(symptoms, profile)

        prompt = self._build_prompt(symptoms, profile)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        
        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=self.cfg.get("max_new_tokens", 64),
                do_sample=True,
                temperature=0.3,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                top_p=0.9
            )
        
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

    def _build_prompt(self, symptoms: list[str], profile: PatientProfile) -> str:
        """Matches the format used during LoRA training."""
        symptom_list = ", ".join(symptoms)
        # We use a similar format to what was in data.jsonl
        instruction = "Act as a physician conducting a clinical interview. Based on the provided symptoms, ask the single most diagnostically useful follow-up question."
        return f"Instruction: {instruction}\nInput: {symptom_list}\nResponse:"

    def _generate_fallback(self, symptoms: list[str], profile: PatientProfile) -> str:
        """Fallback if the LoRA model cannot be loaded."""
        # Simple heuristic fallback
        main_symptom = symptoms[0].lower()
        if "chest" in main_symptom:
            return "Does the pain radiate to your left arm or jaw?"
        if "abdominal" in main_symptom or "stomach" in main_symptom:
            return "When did the pain start, and is it constant or intermittent?"
        return f"Could you tell me more about the onset of your {main_symptom}?"

# Global instance for easy access
interviewer = SymptomInterviewer()

async def next_question(req: InterviewerRequest, llm: Any = None) -> InterviewerResponse:
    """Generates the next clinical question and formats it as an InterviewerResponse."""
    question_text = interviewer.generate_question(req.symptoms, req.patient_profile)
    
    return InterviewerResponse(
        request_id=str(uuid.uuid4()),
        question=question_text,
        rationale="Based on fine-tuned clinical interview patterns.",
        expected_answer_type=AnswerType.FREE_TEXT,
        model_version=interviewer.cfg.get("version", "flan-t5-lora-v1")
    )
