"""Inference for the LightGBM disease classifier (130+ symptoms -> ICD-10)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from pulsepoint_ai.core.config import get_models_config

class DiseaseClassifier:
    def __init__(self) -> None:
        self.cfg = get_models_config()["disease_classifier"]
        self._model = None
        self._label_map = None
        self._feature_names = None

    def _load(self):
        if self._model:
            return

        model_path = Path(self.cfg["artifact"])
        if not model_path.exists():
            print(f"Warning: Disease model not found at {model_path}. Using fallback logic.")
            return

        self._model = lgb.Booster(model_file=str(model_path))
        self._label_map = json.loads(Path(self.cfg["label_map"]).read_text())
        self._feature_names = json.loads(Path(self.cfg["feature_names"]).read_text())
        # Build id -> icd10 mapping. Supports two label_map formats:
        #   {icd10: int}           -> simple
        #   {icd10: {id, name}}    -> rich (training script produces this)
        id_to_label: dict[int, str] = {}
        for key, val in self._label_map.items():
            if isinstance(val, int):
                id_to_label[val] = key
            elif isinstance(val, dict) and "id" in val:
                id_to_label[int(val["id"])] = key
        self._id_to_label = id_to_label

    def predict(
        self,
        symptoms: list[str],
        age: int | None = None,
        gender: str | None = None,
        top_k: int | None = None
    ) -> dict[str, Any]:
        """Predicts Top-K conditions from a list of symptoms."""
        self._load()
        top_k = top_k or self.cfg.get("top_k", 5)

        if not self._model:
            return self._fallback_prediction(symptoms, top_k)

        # 1. Vectorize input (one-hot encoding)
        x = np.zeros((1, len(self._feature_names)), dtype=np.float32)
        for s in symptoms:
            if s in self._feature_names:
                idx = self._feature_names.index(s)
                x[0, idx] = 1.0

        # 2. Inference
        probs = self._model.predict(x)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]

        # 3. Feature importance (shared across predictions for demo)
        importance = self._model.feature_importance(importance_type="gain")
        top_feat_idx = np.argsort(importance)[::-1][:5]
        top_features = [
            {"feature": self._feature_names[i], "shap": float(importance[i])}
            for i in top_feat_idx if x[0, i] > 0
        ]

        # 4. Format results matching DiseasePrediction schema
        predictions = []
        for idx in top_indices:
            predictions.append({
                "icd10": self._id_to_label[idx],
                "name": self._get_condition_name(self._id_to_label[idx]),
                "prob": float(probs[idx]),
                "top_features": top_features,
            })

        return {
            "predictions": predictions,
            "model_version": self.cfg.get("version", "v0.1.0"),
        }

    def _fallback_prediction(self, symptoms: list[str], top_k: int) -> dict[str, Any]:
        """Rule-based fallback when ML model isn't shipped."""
        symptom_set = {s.lower().replace(" ", "_") for s in symptoms}
        # Coarse symptom -> condition prior table
        candidates = [
            ("J45.909", "Asthma", {"shortness_of_breath", "wheezing", "cough"}),
            ("J18.9", "Pneumonia", {"fever", "cough", "shortness_of_breath", "chest_pain"}),
            ("I21.9", "Acute Myocardial Infarction", {"chest_pain", "shortness_of_breath", "sweating"}),
            ("J06.9", "Upper Respiratory Infection", {"sore_throat", "cough", "fever", "runny_nose"}),
            ("R51", "Headache", {"headache"}),
            ("K59.00", "Constipation", {"constipation", "abdominal_pain"}),
            ("E11.9", "Type 2 Diabetes", {"polyuria", "polydipsia", "fatigue"}),
            ("I10", "Hypertension", {"headache", "dizziness"}),
            ("R50.9", "Fever, unspecified", {"fever"}),
            ("J00", "Common Cold", {"runny_nose", "sore_throat", "cough"}),
        ]
        scored = []
        for icd10, name, syms in candidates:
            overlap = len(symptom_set & syms)
            if overlap > 0:
                scored.append((icd10, name, overlap / max(len(syms), 1)))
        scored.sort(key=lambda t: t[2], reverse=True)
        if not scored:
            scored = [("R69", "Unknown/Unspecified", 0.1)]

        total = sum(s[2] for s in scored) or 1.0
        predictions = [
            {
                "icd10": icd10,
                "name": name,
                "prob": round(score / total, 3),
                "top_features": [],
            }
            for icd10, name, score in scored[:top_k]
        ]
        return {"predictions": predictions, "model_version": "fallback_v1"}

    # Built-in ICD-10 -> human name fallback. The label_map.json shipped with
    # the trained classifier is used in preference when available.
    _ICD10_NAME_FALLBACK = {
        "I21.9": "Acute Myocardial Infarction",
        "I26.9": "Pulmonary Embolism",
        "I50.9": "Acute Heart Failure",
        "I10": "Hypertension",
        "J00": "Common Cold",
        "J06.9": "Upper Respiratory Infection",
        "J18.9": "Pneumonia",
        "J45.909": "Asthma",
        "K59.00": "Constipation",
        "E11.9": "Type 2 Diabetes",
        "R51": "Headache",
        "R50.9": "Fever, unspecified",
        "R69": "Unknown/Unspecified",
    }

    def _get_condition_name(self, icd10: str) -> str:
        # Prefer label_map.json values if it embeds a name (some training pipelines
        # store {"icd10": {"id": int, "name": str}}); otherwise use the static map.
        if isinstance(self._label_map, dict):
            entry = self._label_map.get(icd10)
            if isinstance(entry, dict) and entry.get("name"):
                return str(entry["name"])
        return self._ICD10_NAME_FALLBACK.get(icd10, icd10)

# Global instance
disease_classifier = DiseaseClassifier()

# Module-level convenience proxies so callers can do `infer.predict(...)`
def predict(
    symptoms: list[str], 
    age: int | None = None, 
    gender: str | None = None, 
    top_k: int | None = None
) -> dict[str, Any]:
    return disease_classifier.predict(symptoms, age=age, gender=gender, top_k=top_k)
