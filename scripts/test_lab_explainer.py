import asyncio
import os

from pulsepoint_ai.predict.lab_detector import detect_labs
from pulsepoint_ai.predict.lab_explainer import explain_flags

from pulsepoint_ai.core.schemas.common import Gender
from pulsepoint_ai.llm.client import LLMClient


async def main():
    print("🚀 Testing PulsePoint AI Lab Explainer...")


    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: No API keys found in environment. Please set GOOGLE_API_KEY for Gemini.")


    sample_ocr = """
    Patient: John Doe
    Test Results:
    Hemoglobin: 10.5 g/dL
    WBC Count: 15.0 x10^9/L
    Fasting Glucose: 160 mg/dL
    Platelets: 200 x10^9/L
    """

    print("\n--- Step 1: Detecting Labs (Deterministic) ---")

    results, _normal_count = detect_labs(sample_ocr, age=45, gender=Gender.MALE)

    flagged = [f for f in results if f.status != "normal"]
    print(f"Detected {len(results)} labs. Flagged {len(flagged)} abnormal results.")
    for f in flagged:
        print(f"  [!] {f.name}: {f.value} {f.unit} ({f.status})")


    print("\n--- Step 2: Generating AI Explanations ---")
    llm = LLMClient()
    annotated, blocked = await explain_flags(results, llm=llm)

    print("\n--- FINAL RESULTS ---")
    for f in annotated:
        if f.explanation:
            print(f"TEST: {f.name} ({f.status})")
            print(f"EXPLANATION: {f.explanation}")
            print("-" * 30)

    if blocked:
        print(f"Blocked potential hallucinations: {blocked}")

if __name__ == "__main__":
    asyncio.run(main())
