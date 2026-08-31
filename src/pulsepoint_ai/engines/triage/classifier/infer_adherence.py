"""Inference for the LightGBM follow-up adherence/non-adherence classifier."""
from __future__ import annotations

import json
import re

import lightgbm as lgb
import numpy as np

from pulsepoint_ai.core.config import get_settings
from pulsepoint_ai.core.schemas.common import PatientProfile, SeverityTier


class AdherenceClassifier:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_dir = self.settings.models_dir / "adherence_lgbm"
        self._model = None
        self._feature_names = None

    def _load(self):
        if self._model:
            return

        model_path = self.model_dir / "model.txt"
        feat_path = self.model_dir / "feature_names.json"

        if not model_path.exists():
            print(f"Warning: Adherence model not found at {model_path}. Using fallback heuristic.")
            return

        try:
            self._model = lgb.Booster(model_file=str(model_path))
            if feat_path.exists():
                self._feature_names = json.loads(feat_path.read_text())
        except Exception as e:
            print(f"Warning: Failed to load adherence model ({e}). Using fallback heuristic.")
            self._model = None

    def parse_rural_features(
        self,
        profile: PatientProfile,
        symptoms: list[str],
        severity_tier: SeverityTier
    ) -> dict[str, float]:
        """Extracts numerical features for the LightGBM model from screening context."""
        all_text = " ".join(symptoms + profile.known_conditions).lower()


        distance = 5.0
        dist_match = re.search(r"\bdistance[\s:=]*(\d+(?:\.\d+)?)", all_text)
        if dist_match:
            distance = float(dist_match.group(1))
        else:
            km_match = re.search(r"(\d+(?:\.\d+)?)\s*km", all_text)
            if km_match:
                distance = float(km_match.group(1))
            elif "far from phc" in all_text or "remote village" in all_text:
                distance = 15.0
            elif "very far" in all_text:
                distance = 25.0


        occupation_farming = 0.0
        occupation_laborer = 0.0
        if any(x in all_text for x in ["farming", "farmer", "agriculture", "khet", "crop", "field"]):
            occupation_farming = 1.0
        elif any(x in all_text for x in ["laborer", "manual labor", "daily wage", "construction", "mazdoor"]):
            occupation_laborer = 1.0


        tobacco_alcohol = 0.0
        if any(x in all_text for x in ["tobacco", "bidi", "smoke", "smoking", "alcohol", "drinking", "beer", "daroo"]):
            tobacco_alcohol = 1.0


        severity_map = {
            SeverityTier.LOW: 0.0,
            SeverityTier.MEDIUM: 1.0,
            SeverityTier.HIGH: 2.0,
            SeverityTier.URGENT: 3.0,
            SeverityTier.EMERGENCY: 4.0
        }
        severity_val = severity_map.get(severity_tier, 0.0)


        previous_adherence = 1.0
        if any(x in all_text for x in ["non-adherence", "non adherence", "missed screening", "defaulted"]):
            previous_adherence = 0.0

        return {
            "age": float(profile.age),
            "distance_to_phc": distance,
            "occupation_farming": occupation_farming,
            "occupation_laborer": occupation_laborer,
            "tobacco_alcohol_usage": tobacco_alcohol,
            "severity_tier_val": severity_val,
            "previous_adherence": previous_adherence
        }

    def predict_default_risk(
        self,
        profile: PatientProfile,
        symptoms: list[str],
        severity_tier: SeverityTier
    ) -> float:
        """Predicts the probability of the patient defaulting on follow-up (0.0 to 1.0)."""
        self._load()

        feats = self.parse_rural_features(profile, symptoms, severity_tier)

        if not self._model or not self._feature_names:

            z = (
                -1.2
                + 0.12 * feats["distance_to_phc"]
                + 0.65 * feats["occupation_farming"]
                + 0.80 * feats["occupation_laborer"]
                + 0.35 * feats["tobacco_alcohol_usage"]
                - 0.55 * feats["severity_tier_val"]
                - 1.50 * feats["previous_adherence"]
                + 0.008 * (feats["age"] - 45)
            )
            prob = 1.0 / (1.0 + np.exp(-z))
            return float(prob)


        x = np.zeros((1, len(self._feature_names)), dtype=np.float32)
        for i, name in enumerate(self._feature_names):
            x[0, i] = feats.get(name, 0.0)

        try:
            prob = self._model.predict(x)[0]
            return float(prob)
        except Exception as e:
            print(f"Error during adherence LightGBM inference: {e}. Using fallback.")

            z = (
                -1.2
                + 0.12 * feats["distance_to_phc"]
                + 0.65 * feats["occupation_farming"]
                + 0.80 * feats["occupation_laborer"]
                + 0.35 * feats["tobacco_alcohol_usage"]
                - 0.55 * feats["severity_tier_val"]
                - 1.50 * feats["previous_adherence"]
                + 0.008 * (feats["age"] - 45)
            )
            return float(1.0 / (1.0 + np.exp(-z)))


adherence_classifier = AdherenceClassifier()

def predict(
    profile: PatientProfile,
    symptoms: list[str],
    severity_tier: SeverityTier
) -> float:
    return adherence_classifier.predict_default_risk(profile, symptoms, severity_tier)


def score_adherence_risk(
    patient: dict | PatientProfile,
    symptoms: list[str] | None = None,
    severity_tier: SeverityTier = SeverityTier.LOW
) -> float:
    """Scores adherence risk for a patient dictionary or PatientProfile."""
    if isinstance(patient, dict):
        raw_dist = patient.get("distance_to_phc_km")
        if raw_dist is None:
            raw_dist = patient.get("distance_to_phc", 5.0)
        distance = float(raw_dist) if raw_dist is not None else 5.0

        farming = 1.0 if patient.get("occupation") == "farming" or patient.get("farmer") else 0.0
        laborer = 1.0 if patient.get("daily_wage_earner") or patient.get("laborer") or patient.get("occupation") == "laborer" else 0.0
        tobacco_alcohol = 1.0 if patient.get("tobacco_alcohol_usage") or patient.get("substance_use") else 0.0

        sev = patient.get("severity", severity_tier)
        severity_map = {
            SeverityTier.LOW: 0.0,
            SeverityTier.MEDIUM: 1.0,
            SeverityTier.HIGH: 2.0,
            SeverityTier.URGENT: 3.0,
            SeverityTier.EMERGENCY: 4.0
        }
        severity_val = severity_map.get(sev, 0.0) if isinstance(sev, SeverityTier) else float(sev or 0.0)

        prev_default = patient.get("previous_default", False)
        prev_adh = patient.get("previous_adherence")
        if prev_adh is not None:
            previous_adherence = float(prev_adh)
        elif prev_default:
            previous_adherence = 0.0
        else:
            previous_adherence = 1.0

        raw_age = patient.get("age", 45)
        age = float(raw_age) if raw_age is not None else 45.0

        feats = {
            "age": age,
            "distance_to_phc": distance,
            "occupation_farming": farming,
            "occupation_laborer": laborer,
            "tobacco_alcohol_usage": tobacco_alcohol,
            "severity_tier_val": severity_val,
            "previous_adherence": previous_adherence
        }

        adherence_classifier._load()
        if not adherence_classifier._model or not adherence_classifier._feature_names:
            z = (
                -1.2
                + 0.12 * feats["distance_to_phc"]
                + 0.65 * feats["occupation_farming"]
                + 0.80 * feats["occupation_laborer"]
                + 0.35 * feats["tobacco_alcohol_usage"]
                - 0.55 * feats["severity_tier_val"]
                - 1.50 * feats["previous_adherence"]
                + 0.008 * (feats["age"] - 45)
            )
            return float(1.0 / (1.0 + np.exp(-z)))

        x = np.zeros((1, len(adherence_classifier._feature_names)), dtype=np.float32)
        for i, name in enumerate(adherence_classifier._feature_names):
            x[0, i] = feats.get(name, 0.0)

        try:
            return float(adherence_classifier._model.predict(x)[0])
        except Exception:
            z = (
                -1.2
                + 0.12 * feats["distance_to_phc"]
                + 0.65 * feats["occupation_farming"]
                + 0.80 * feats["occupation_laborer"]
                + 0.35 * feats["tobacco_alcohol_usage"]
                - 0.55 * feats["severity_tier_val"]
                - 1.50 * feats["previous_adherence"]
                + 0.008 * (feats["age"] - 45)
            )
            return float(1.0 / (1.0 + np.exp(-z)))

    return adherence_classifier.predict_default_risk(patient, symptoms or [], severity_tier)

