import sys
import os

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import re
from pulsepoint_ai.core.config import get_lab_ranges
from pulsepoint_ai.engines.predict.lab_detector import detect_labs

ocr_text = """
    Patient Report
    Fasting Blood Sugar: 128 mg/dL
    HB-A1C = 6.8 %
    microalbuminuria: 35 mg/L
    Random Glucose level is 145 mg/dL
    Total Cholesterol : 185
    Urine Protein - 25 mg/dL
"""

text = " ".join(ocr_text.split()).lower()
print(f"Normalized Text: {repr(text)}")

cfg = get_lab_ranges()["tests"]
for canonical, test_cfg in cfg.items():
    if canonical in ["fasting_glucose", "random_glucose", "hba1c"]:
        print(f"\n--- {canonical} ---")
        for alias in test_cfg["aliases"]:
            cleaned_alias = alias.lower()
            cleaned_alias = re.sub(r"[\s\-_.]+", "###PLACEHOLDER###", cleaned_alias)
            
            alias_pattern = re.escape(cleaned_alias)
            alias_pattern = alias_pattern.replace("###PLACEHOLDER###", r"[\s\-_.]*")
            
            pattern = re.compile(
                rf"\b{alias_pattern}\b[^\d]{{0,25}}?([-+]?\d+(?:\.\d+)?)", re.IGNORECASE
            )
            m = pattern.search(text)
            print(f"Alias: {repr(alias)} | Pattern: {repr(pattern.pattern)} | Match: {m}")
