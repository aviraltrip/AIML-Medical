import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

MODEL_PATH = "models/interviewer_lora"
BASE_MODEL = "google/flan-t5-small"

def test_inference():
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Load base model
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    model.eval()

    # Test cases
    test_cases = [
        {
            "instruction": "Given the patient's reported symptoms of chest pain, identify the most diagnostically useful follow-up question.",
            "input": ["Sharp chest pain", "worse when breathing", "shortness of breath"]
        },
        {
            "instruction": "Act as a physician conducting a clinical interview. Based on the provided symptoms, ask the single most diagnostically useful follow-up question.",
            "input": ["right lower quadrant pain", "fever", "nausea"]
        }
    ]

    print("\nRunning Tests:")
    print("-" * 30)
    
    for case in test_cases:
        inp = ", ".join(case["input"]) if isinstance(case["input"], list) else case["input"]
        prompt = f"Instruction: {case['instruction']}\nInput: {inp}\nResponse:"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=64,
                do_sample=True,
                temperature=0.3,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                top_p=0.9
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Symptoms: {inp}")
        print(f"Model Question: {response}")
        print("-" * 30)

if __name__ == "__main__":
    test_inference()
