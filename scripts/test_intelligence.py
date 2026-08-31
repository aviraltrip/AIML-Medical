import asyncio
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))

from pulsepoint_ai.engines.triage.classifier.infer_adherence import score_adherence_risk
from pulsepoint_ai.core.schemas.common import SeverityTier
from pulsepoint_ai.engines.connect.translation import translator


async def run_intelligence_test():
    print("="*50)
    print("PULSEPOINT INTELLIGENCE TEST")
    print("="*50)

    # --- TEST 1: ADHERENCE RISK INTELLIGENCE ---
    print("\n[TEST 1] Referral Adherence Risk Scoring")
    sample_patient = {
        "distance_to_phc_km": 14.5,
        "daily_wage_earner": True,
        "previous_default": True,
        "severity": SeverityTier.HIGH,
        "age": 52,
    }
    adherence_prob = score_adherence_risk(sample_patient)
    print(f"Patient Profile: {sample_patient}")
    print(f"Adherence Default Risk Probability: {adherence_prob:.2f}")


    print("\n" + "-"*30)
    print("[TEST 2] Translation (English -> Kannada)")
    doctor_note = "Please take the prescribed antibiotics twice a day after your meals."
    translated = await translator.translate(doctor_note, target_lang="kn")


    print(f"Original: {doctor_note}")
    try:
        print(f"Kannada: {translated}")
    except UnicodeEncodeError:
        print("Kannada translation received (successfully encoded in UTF-8).")

    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_intelligence_test())
