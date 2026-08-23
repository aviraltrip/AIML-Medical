"""Inference for the LightGBM triage severity classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from pulsepoint_ai.core.config import get_models_config
from pulsepoint_ai.core.schemas.common import SeverityTier, Vitals

_DEFAULT_TIER_ORDER = [
    SeverityTier.LOW,
    SeverityTier.MEDIUM,
    SeverityTier.HIGH,
    SeverityTier.URGENT,
    SeverityTier.EMERGENCY,
]



_VITAL_FEATURE_ALIASES = {
    "pulse_bpm": ("pulse_bpm", "heart_rate", "hr"),
    "bp_systolic": ("bp_systolic", "sbp", "systolic_bp"),
    "bp_diastolic": ("bp_diastolic", "dbp", "diastolic_bp"),
    "spo2": ("spo2", "oxygen_saturation"),
    "temp_c": ("temp_c", "temperature_c", "temperature"),
    "respiratory_rate": ("respiratory_rate", "rr", "resp_rate"),
    "blood_sugar_mg_dl": ("blood_sugar_mg_dl", "glucose", "blood_glucose"),
}


class TriageClassifier:
    def __init__(self) -> None:
        self.cfg = get_models_config()["triage_classifier"]
        self._model = None
        self._feature_names: list[str] | None = None
        self._tiers: list[SeverityTier] = _DEFAULT_TIER_ORDER

    def _load(self):
        if self._model:
            return

        model_path = Path(self.cfg["artifact"])
        if not model_path.exists():
            print(f"Warning: Triage model not found at {model_path}. Using fallback logic.")
            return

        try:
            self._model = lgb.Booster(model_file=str(model_path))
            feat_path = Path(self.cfg["feature_names"])
            if feat_path.exists():
                self._feature_names = json.loads(feat_path.read_text())


            label_map_path = Path(self.cfg.get("label_map", "")) if self.cfg.get("label_map") else None
            if label_map_path and label_map_path.exists():
                label_map = json.loads(label_map_path.read_text())

                ordered = sorted(label_map.items(), key=lambda kv: kv[1])
                resolved = []
                for name, _ in ordered:
                    try:
                        resolved.append(SeverityTier(name.upper()))
                    except ValueError:
                        resolved.append(SeverityTier.MEDIUM)
                if resolved:
                    self._tiers = resolved
        except Exception as e:
            print(f"Warning: Failed to load triage model ({e}). Using fallback logic.")
            self._model = None

    def predict(
        self,
        symptoms: list[str],
        vitals: Vitals,
        patient_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Predicts severity tier from symptoms and vitals."""
        self._load()

        if not self._model or not self._feature_names:
            return self._fallback_prediction(symptoms)


        x = np.zeros((1, len(self._feature_names)), dtype=np.float32)


        for s in symptoms:
            for variant in (s, s.lower(), s.lower().replace(" ", "_")):
                if variant in self._feature_names:
                    x[0, self._feature_names.index(variant)] = 1.0
                    break


        for vitals_attr, aliases in _VITAL_FEATURE_ALIASES.items():
            value = getattr(vitals, vitals_attr, None)
            if value is None:
                continue
            for alias in aliases:
                if alias in self._feature_names:
                    x[0, self._feature_names.index(alias)] = float(value)
                    break


        try:
            probs = self._model.predict(x)[0]
        except Exception as e:
            print(f"Triage inference failed ({e}). Falling back.")
            return self._fallback_prediction(symptoms)

        probs = np.atleast_1d(np.asarray(probs, dtype=np.float64))


        tiers = self._tiers[: len(probs)] or _DEFAULT_TIER_ORDER[: len(probs)]
        if len(tiers) < len(probs):
            tiers = list(tiers) + _DEFAULT_TIER_ORDER[len(tiers) : len(probs)]

        tier_idx = int(np.argmax(probs))
        predicted_tier = tiers[tier_idx]

        return {
            "tier": predicted_tier,
            "probs": {t.value: float(probs[i]) for i, t in enumerate(tiers)},
            "feature_importance": self._get_top_features(x),
            "model_version": self.cfg.get("version", "v0.1.0"),
        }

    def _get_top_features(self, x: np.ndarray) -> list[dict[str, Any]]:
        """Return list of {feature, shap} matching the FeatureImportance schema."""
        if not self._model or not self._feature_names:
            return []
        importance = self._model.feature_importance(importance_type="gain")
        top_idx = np.argsort(importance)[::-1][:5]
        return [
            {"feature": self._feature_names[i], "shap": float(importance[i])}
            for i in top_idx
            if x[0, i] > 0
        ]

    def _fallback_prediction(self, symptoms: list[str]) -> dict[str, Any]:
        """Rule-based fallback if the ML model isn't trained yet."""
        normalized = {s.lower().replace(" ", "_") for s in symptoms}
        severity = SeverityTier.LOW
        if normalized & {"chest_pain", "shortness_of_breath", "stroke_symptoms", "unconscious"}:
            severity = SeverityTier.HIGH
        elif normalized & {"high_fever", "severe_headache", "vomiting_blood"}:
            severity = SeverityTier.MEDIUM
        return {
            "tier": severity,
            "probs": {t.value: 0.2 for t in SeverityTier},
            "feature_importance": [],
            "model_version": "fallback_v1",
        }



triage_classifier = TriageClassifier()


def predict(
    symptoms: list[str],
    vitals: Vitals,
    patient_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return triage_classifier.predict(symptoms, vitals, patient_profile)
