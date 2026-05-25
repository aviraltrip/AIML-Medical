"""Inference for the LightGBM follow-up adherence/non-adherence classifier."""
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any
import lightgbm as lgb
import numpy as np

from pulsepoint_ai.core.config import get_models_config, get_settings
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
        
        # 1. Parse distance_to_phc (e.g. "12 km", "distance: 8.5", "far from phc")
        distance = 5.0  # default rural baseline
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

        # 2. Parse occupations
        occupation_farming = 0.0
        occupation_laborer = 0.0
        if any(x in all_text for x in ["farming", "farmer", "agriculture", "khet", "crop", "field"]):
            occupation_farming = 1.0
        elif any(x in all_text for x in ["laborer", "manual labor", "daily wage", "construction", "mazdoor"]):
            occupation_laborer = 1.0

        # 3. Parse tobacco and alcohol
        tobacco_alcohol = 0.0
        if any(x in all_text for x in ["tobacco", "bidi", "smoke", "smoking", "alcohol", "drinking", "beer", "daroo"]):
            tobacco_alcohol = 1.0

        # 4. Map severity tier to numerical value (0 to 4)
        severity_map = {
            SeverityTier.LOW: 0.0,
            SeverityTier.MEDIUM: 1.0,
            SeverityTier.HIGH: 2.0,
            SeverityTier.URGENT: 3.0,
            SeverityTier.EMERGENCY: 4.0
        }
        severity_val = severity_map.get(severity_tier, 0.0)

        # 5. Parse previous adherence history
        previous_adherence = 1.0  # Default to clean history
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
            # Fallback heuristic calculation matching the training model's weights
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
            
        # Compile vector aligned with features list
        x = np.zeros((1, len(self._feature_names)), dtype=np.float32)
        for i, name in enumerate(self._feature_names):
            x[0, i] = feats.get(name, 0.0)
            
        try:
            prob = self._model.predict(x)[0]
            return float(prob)
        except Exception as e:
            print(f"Error during adherence LightGBM inference: {e}. Using fallback.")
            # Duplicate fallback
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

# Global instance
adherence_classifier = AdherenceClassifier()

def predict(
    profile: PatientProfile,
    symptoms: list[str],
    severity_tier: SeverityTier
) -> float:
    return adherence_classifier.predict_default_risk(profile, symptoms, severity_tier)
