"""Inference for the LightGBM disease classifier (130+ symptoms -> ICD-10)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import lightgbm as lgb

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
        """Predicts Top-K chronic conditions (Diabetes & Hypertension) from symptoms, age, and gender."""
        from pulsepoint_ai.core.schemas.common import Gender, PatientProfile, Vitals
        from pulsepoint_ai.engines.chronic_scoring import evaluate_chronic_risk

        age_val = age if age is not None else 45
        gender_enum = Gender.MALE if gender == "male" else (Gender.FEMALE if gender == "female" else Gender.OTHER)

        profile = PatientProfile(
            age=age_val,
            gender=gender_enum,
            known_conditions=[],
            medications=[],
            allergies=[]
        )
        vitals = Vitals()


        risk_results = evaluate_chronic_risk(profile, vitals, symptoms)

        predictions = [
            {
                "icd10": "E11",
                "name": "Type 2 Diabetes Risk",
                "prob": round(float(risk_results["diabetes_prob"]), 2),
                "top_features": [
                    {"feature": "Age Factor", "shap": float(risk_results["diabetes_breakdown"]["age_score"])},
                    {"feature": "Waist Circumference", "shap": float(risk_results["diabetes_breakdown"]["waist_score"])},
                    {"feature": "Physical Inactivity", "shap": float(risk_results["diabetes_breakdown"]["activity_score"])},
                    {"feature": "Family History", "shap": float(risk_results["diabetes_breakdown"]["family_score"])},
                ]
            },
            {
                "icd10": "I10",
                "name": "Hypertension Risk",
                "prob": round(float(risk_results["hypertension_prob"]), 2),
                "top_features": [
                    {"feature": "Somatic headache / dizziness symptoms", "shap": 20.0 if any(x in " ".join(symptoms).lower() for x in ["headache", "dizziness"]) else 0.0},
                    {"feature": "Behavioral tobacco/alcohol risk", "shap": 15.0 if any(x in " ".join(symptoms).lower() for x in ["tobacco", "alcohol", "smoking", "bidi"]) else 0.0},
                    {"feature": "Age Factor", "shap": float(risk_results["diabetes_breakdown"]["age_score"])},
                ]
            }
        ]


        if risk_results["diabetes_breakdown"].get("waist_score", 0) >= 20:
            predictions.append({
                "icd10": "E66.9",
                "name": "Metabolic Syndrome Risk / Obesity",
                "prob": 0.75,
                "top_features": [
                    {"feature": "Waist Circumference", "shap": 20.0}
                ]
            })

        predictions.sort(key=lambda x: x["prob"], reverse=True)
        top_k = top_k or self.cfg.get("top_k", 5)

        return {
            "predictions": predictions[:top_k],
            "model_version": "chronic_scoring_v1.0.0",
        }

    def _fallback_prediction(self, symptoms: list[str], top_k: int) -> dict[str, Any]:
        """Rule-based fallback when ML model isn't shipped."""
        symptom_set = {s.lower().replace(" ", "_") for s in symptoms}

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



    _ICD10_NAME_FALLBACK: ClassVar[dict[str, str]] = {
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


        if isinstance(self._label_map, dict):
            entry = self._label_map.get(icd10)
            if isinstance(entry, dict) and entry.get("name"):
                return str(entry["name"])
        return self._ICD10_NAME_FALLBACK.get(icd10, icd10)


disease_classifier = DiseaseClassifier()


def predict(
    symptoms: list[str],
    age: int | None = None,
    gender: str | None = None,
    top_k: int | None = None
) -> dict[str, Any]:
    return disease_classifier.predict(symptoms, age=age, gender=gender, top_k=top_k)
