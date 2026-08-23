import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tests.test_chronic_refactor import (
    test_adherence_classifier,
    test_api_endpoints_contract,
    test_chronic_risk_evaluation,
    test_fuzzy_ocr_lab_parser,
    test_hypertension_staging,
    test_idrs_calculations,
    test_parameter_inferences,
)

if __name__ == "__main__":
    print("Starting PulsePoint Chronic Refactor Unit Tests...")
    try:
        print("[1/7] Running IDRS Points Calculation Tests...")
        test_idrs_calculations()
        print("IDRS points calculation: PASSED")

        print("[2/7] Running Parameter Clues Inference Tests...")
        test_parameter_inferences()
        print("Parameter clues inference: PASSED")

        print("[3/7] Running JNC-8 Hypertension Staging Tests...")
        test_hypertension_staging()
        print("JNC-8 hypertension staging: PASSED")

        print("[4/7] Running Unified Chronic Staging & Rules Tests...")
        test_chronic_risk_evaluation()
        print("Unified chronic staging and rules: PASSED")

        print("[5/7] Running Fuzzy OCR Lab Parser Tests...")
        test_fuzzy_ocr_lab_parser()
        print("Fuzzy OCR lab parser: PASSED")

        print("[6/7] Running LightGBM Referral Adherence Classifier Tests...")
        test_adherence_classifier()
        print("LightGBM referral adherence classifier: PASSED")

        print("[7/7] Running API Response Schema & Payloads Tests...")
        test_api_endpoints_contract()
        print("API response schema and payload compatibility: PASSED")

        print("\nALL TESTS PASSED SUCCESSFULLY! BACKEND REFIT IS COMPATIBLE & SAFE.")
        sys.exit(0)
    except AssertionError:
        print("\nTEST ASSERTION FAILURE:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception:
        print("\nTEST RUNNER EXECUTION ERROR:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
