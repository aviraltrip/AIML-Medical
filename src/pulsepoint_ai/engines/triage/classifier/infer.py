"""Inference for the LightGBM triage severity classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from pulsepoint_ai.core.config import get_models_config
from pulsepoint_ai.core.schemas.common import SeverityTier, Vitals

class TriageClassifier:
    def __init__(self) -> None:
        self.cfg = get_models_config()["triage_classifier"]
        self._model = None
        self._feature_names = None

    def _load(self):
        if self._model:
            return
        
        model_path = Path(self.cfg["artifact"])
        if not model_path.exists():
            # For hackathon demo, we handle the case where the model hasn't been trained yet
            print(f"Warning: Triage model not found at {model_path}. Using fallback logic.")
            return

        self._model = lgb.Booster(model_file=str(model_path))
        self._feature_names = json.loads(Path(self.cfg["feature_names"]).read_text())

    def predict(self, symptoms: list[str], vitals: Vitals, patient_profile: dict[str, Any] | None = None) -> dict[str, Any]:
        """Predicts severity tier from symptoms and vitals."""
        self._load()
        
        if not self._model:
            return self._fallback_prediction(symptoms)

        # 1. Feature Engineering
        x = np.zeros((1, len(self._feature_names)), dtype=np.float32)
        # Add symptoms
        for s in symptoms:
            if s in self._feature_names:
                x[0, self._feature_names.index(s)] = 1.0
        # Add vitals
        if "heart_rate" in self._feature_names:
            x[0, self._feature_names.index("heart_rate")] = vitals.heart_rate
        if "bp_systolic" in self._feature_names:
            x[0, self._feature_names.index("bp_systolic")] = vitals.bp_systolic
        
        # 2. Inference
        probs = self._model.predict(x)[0]
        tier_idx = np.argmax(probs)
        
        # Mapping (Depends on how you labeled your training data)
        tiers = [SeverityTier.LOW, SeverityTier.MEDIUM, SeverityTier.HIGH, SeverityTier.CRITICAL]
        predicted_tier = tiers[tier_idx]
        
        return {
            "tier": predicted_tier,
            "probs": {t.value: float(probs[i]) for i, t in enumerate(tiers)},
            "feature_importance": self._get_top_features(x),
            "model_version": self.cfg.get("version", "v0.1.0")
        }

    def _get_top_features(self, x: np.ndarray) -> list[str]:
        if not self._model: return []
        importance = self._model.feature_importance(importance_type="gain")
        top_idx = np.argsort(importance)[::-1][:5]
        return [self._feature_names[i] for i in top_idx if x[0, i] > 0]

    def _fallback_prediction(self, symptoms: list[str]) -> dict[str, Any]:
        """Rule-based fallback if the ML model isn't trained yet."""
        severity = SeverityTier.LOW
        if any(s in ["chest_pain", "shortness_of_breath"] for s in symptoms):
            severity = SeverityTier.HIGH
        return {
            "tier": severity,
            "probs": {t.value: 0.25 for t in SeverityTier},
            "feature_importance": ["chest_pain"] if severity == SeverityTier.HIGH else [],
            "model_version": "fallback_v1"
        }

# Global instance
triage_classifier = TriageClassifier()

# Module-level convenience proxies so callers can do `infer.predict(...)`
def predict(symptoms: list[str], vitals: Vitals, patient_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    return triage_classifier.predict(symptoms, vitals, patient_profile)
