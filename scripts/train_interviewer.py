import json
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# Configuration
DATA_PATH = "data/processed/interviewer_train.jsonl"
MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = "models/interviewer_lora"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 5e-4

def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess_function(examples, tokenizer):
    inputs = []
    for i in range(len(examples["instruction"])):
        # Handle input being a list or string
        inp = examples["input"][i]
        if isinstance(inp, list):
            inp = ", ".join(inp)
        
        prompt = f"Instruction: {examples['instruction'][i]}\nInput: {inp}\nResponse:"
        inputs.append(prompt)

    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    
    # Tokenize targets using text_target
    labels = tokenizer(text_target=examples["output"], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train():
    print(f"Loading data from {DATA_PATH}...")
    raw_data = load_jsonl(DATA_PATH)
    
    # Normalize data: Ensure 'input' is always a string to prevent pyarrow errors
    for item in raw_data:
        if isinstance(item.get("input"), list):
            item["input"] = ", ".join(item["input"])
        elif item.get("input") is None:
            item["input"] = ""

    dataset = Dataset.from_list(raw_data)
    
    print(f"Initializing model and tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT
    )
    
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True,
        remove_columns=dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir="./tmp_trainer",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_strategy="no", 
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete!")

if __name__ == "__main__":
    train()
