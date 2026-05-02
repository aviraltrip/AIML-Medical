"""LoRA fine-tune of Llama-3.1-8B-Instruct for clinical interview question generation.

Designed to run on Google Colab T4 (free tier) with 4-bit quantization.
Mirror this file in notebooks/03_lora_finetune_colab.ipynb for the actual training run.

CLI (only runs locally if GPU available):
    uv run python -m pulsepoint_ai.interview.train_lora \
        --train data/interview_qa/train.jsonl \
        --val data/interview_qa/val.jsonl \
        --out models/interviewer_lora
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def build_chat_text(record: dict) -> str:
    """Convert a Q&A record into Llama-3 chat format."""
    sys = (
        "You are PulsePoint's clinical interviewer. Given current symptoms, "
        "produce the single most diagnostically useful next question as JSON."
    )
    user = json.dumps(
        {
            "symptoms": record["symptoms"],
            "answered": record.get("answered", []),
            "patient_profile": record["patient_profile"],
        }
    )
    assistant = json.dumps(
        {
            "question": record["ideal_next_question"],
            "rationale": record.get("rationale", ""),
            "expected_answer_type": record.get("expected_answer_type", "yes_no"),
        }
    )
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    # Imports deferred so this module is importable without GPU deps installed.
    from datasets import load_dataset  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    ds = load_dataset(
        "json", data_files={"train": str(args.train), "val": str(args.val)}
    )
    ds = ds.map(lambda r: {"text": build_chat_text(r)})

    cfg = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        max_seq_length=1024,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, train_dataset=ds["train"], eval_dataset=ds["val"], args=cfg)
    trainer.train()
    trainer.model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
