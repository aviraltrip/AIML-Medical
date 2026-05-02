"""LoRA-adapter inference for the symptom interviewer."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pulsepoint_ai.core.config import get_models_config
from pulsepoint_ai.core.schemas.common import PatientProfile
from pulsepoint_ai.llm.parsers import extract_json


@lru_cache(maxsize=1)
def _load() -> tuple[Any, Any]:
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    cfg = get_models_config()["interviewer"]
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        load_in_4bit=cfg.get("load_in_4bit", True),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, cfg["adapter"])
    model.eval()
    return tok, model


def generate_next_question(
    symptoms: list[str],
    answered: list[dict],
    patient_profile: PatientProfile,
) -> dict:
    cfg = get_models_config()["interviewer"]
    tok, model = _load()
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are PulsePoint's clinical interviewer. Output strict JSON only.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        + json.dumps(
            {
                "symptoms": symptoms,
                "answered": answered,
                "patient_profile": patient_profile.model_dump(),
            }
        )
        + "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=cfg.get("max_new_tokens", 128),
        do_sample=False,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return extract_json(text)
