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
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
            
        self._model = lgb.Booster(model_file=str(model_path))
        self._label_map = json.loads(Path(self.cfg["label_map"]).read_text())
        self._feature_names = json.loads(Path(self.cfg["feature_names"]).read_text())
        self._id_to_label = {v: k for k, v in self._label_map.items()}

    def predict(
        self, 
        symptoms: list[str], 
        age: int | None = None, 
        gender: str | None = None, 
        top_k: int | None = None
    ) -> dict[str, Any]:
        """Predicts Top-K conditions from a list of symptoms."""
        self._load()
        
        # 1. Vectorize input (one-hot encoding)
        x = np.zeros((1, len(self._feature_names)), dtype=np.float32)
        for s in symptoms:
            if s in self._feature_names:
                idx = self._feature_names.index(s)
                x[0, idx] = 1.0
        
        # 2. Inference
        probs = self._model.predict(x)[0]
        top_k = top_k or self.cfg.get("top_k", 5)
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        # 3. Format results
        results = []
        for idx in top_indices:
            results.append({
                "icd10": self._id_to_label[idx],
                "name": self._get_condition_name(self._id_to_label[idx]),
                "probability": float(probs[idx])
            })
            
        # 4. Feature Importance (Simplified for demo)
        # In a full build, we'd use SHAP here.
        importance = self._model.feature_importance(importance_type="gain")
        top_feat_idx = np.argsort(importance)[::-1][:5]
        top_features = [self._feature_names[i] for i in top_feat_idx if x[0, i] > 0]

        return {
            "top_conditions": results,
            "driving_symptoms": top_features,
            "model_version": self.cfg.get("version", "v0.1.0")
        }

    def _get_condition_name(self, icd10: str) -> str:
        # Placeholder for ICD-10 to Name mapping
        # In production, this would use a full medical dictionary
        mapping = {"I21.9": "Acute Myocardial Infarction", "J18.9": "Pneumonia"}
        return mapping.get(icd10, f"Condition {icd10}")

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
