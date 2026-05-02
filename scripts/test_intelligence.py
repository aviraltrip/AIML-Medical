import os
import sys
import asyncio

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from pulsepoint_ai.engines.predict.trends import analyze_trend, TrendDirection
from pulsepoint_ai.engines.connect.translation import translator

async def run_intelligence_test():
    print("="*50)
    print("PULSEPOINT INTELLIGENCE TEST")
    print("="*50)

    # --- TEST 1: TREND INTELLIGENCE ---
    print("\n[TEST 1] Trend Intelligence (Blood Sugar)")
    # Simulating a worsening trend (Sugar rising: 95 -> 105 -> 120 -> 140)
    sugar_data = [95, 105, 120, 140]
    result = analyze_trend("Blood Sugar", sugar_data, higher_is_better=False)
    
    print(f"Data: {sugar_data}")
    print(f"Result: {result.direction.upper()}")
    print(f"Message: {result.message}")

    # --- TEST 2: TRANSLATION ---
    print("\n" + "-"*30)
    print("[TEST 2] Translation (English -> Kannada)")
    doctor_note = "Please take the prescribed antibiotics twice a day after your meals."
    translated = await translator.translate(doctor_note, target_lang="kn")
    
    # We use a safe print for the translated text to avoid potential encoding issues in some consoles
    print(f"Original: {doctor_note}")
    try:
        print(f"Kannada: {translated}")
    except UnicodeEncodeError:
        print("Kannada translation received (successfully encoded in UTF-8).")
    
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_intelligence_test())
