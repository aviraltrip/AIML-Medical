import json
import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

def generate_medical_data():
    print("PulsePoint: Generating Synthetic Medical Dataset (Native SDK)...")
    
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env")
        return

    genai.configure(api_key=api_key)
    # Using the model discovered in diagnostics
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    cases = [
        "Chest Pain", "Abdominal Pain", "Headache", "Fever", "Cough", 
        "Dizziness", "Shortness of Breath", "Back Pain", "Skin Rash"
    ]
    
    output_path = Path("data/processed/interviewer_train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for case in cases:
            print(f"Generating samples for: {case}...")
            prompt = (
                f"You are a medical professor. Generate 50 clinical interview samples for '{case}'.\n"
                "Format: JSONL with 'instruction', 'input', and 'output'.\n"
                "'input' should be a list of symptoms.\n"
                "'output' should be the single most diagnostically useful follow-up question.\n"
                "Each sample should be on a new line as a valid JSON object.\n"
                "Do not include markdown backticks like ```json."
            )
            
            try:
                response = model.generate_content(prompt)
                lines = response.text.strip().split("\n")
                for line in lines:
                    if not line.strip(): continue
                    try:
                        sample = json.loads(line)
                        # Add the 'text' field required by SFTTrainer
                        sample["text"] = f"### Instruction: {sample['instruction']}\n### Input: {sample['input']}\n### Response: {sample['output']}"
                        f.write(json.dumps(sample) + "\n")
                    except Exception as e:
                        print(f"Skipping invalid line: {e}")
            except Exception as e:
                print(f"Error generating for {case}: {e}")
                
    print(f"Dataset generated at {output_path}")

if __name__ == "__main__":
    generate_medical_data()
