"""Train the LightGBM disease classifier.

Mirrors the structure and label space of the popular Kaggle 'Disease Symptom
Prediction' dataset (41 diseases x 132 symptoms x ~5000 samples). The dataset
is generated locally from a curated disease->symptoms mapping so this script
runs without any Kaggle credentials or network access.

Outputs (all written to models/disease_lgbm/):
    model.txt           - LightGBM Booster
    feature_names.json  - ordered list of 132 symptom feature names
    label_map.json      - {icd10: {id: int, name: str}} consumed by infer.py

Run:
    python scripts/train_disease_classifier.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np

OUTPUT_DIR = Path("models/disease_lgbm")
RNG_SEED = 42
N_SAMPLES_PER_DISEASE = 120
NOISE_PROB_OFF = 0.10
NOISE_PROB_ON = 0.02




DISEASES: dict[str, dict] = {
    "Fungal infection": {
        "icd10": "B49",
        "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"],
    },
    "Allergy": {
        "icd10": "T78.40",
        "symptoms": ["continuous_sneezing", "shivering", "chills", "watering_from_eyes"],
    },
    "GERD": {
        "icd10": "K21.9",
        "symptoms": ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"],
    },
    "Chronic cholestasis": {
        "icd10": "K71.0",
        "symptoms": ["itching", "vomiting", "yellowish_skin", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes"],
    },
    "Drug Reaction": {
        "icd10": "T88.7",
        "symptoms": ["itching", "skin_rash", "stomach_pain", "burning_micturition", "spotting_urination"],
    },
    "Peptic ulcer disease": {
        "icd10": "K27",
        "symptoms": ["vomiting", "indigestion", "loss_of_appetite", "abdominal_pain", "passage_of_gases", "internal_itching"],
    },
    "AIDS": {
        "icd10": "B20",
        "symptoms": ["muscle_wasting", "patches_in_throat", "high_fever", "extra_marital_contacts"],
    },
    "Diabetes": {
        "icd10": "E11.9",
        "symptoms": ["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "blurred_and_distorted_vision", "obesity", "excessive_hunger", "increased_appetite", "polyuria"],
    },
    "Gastroenteritis": {
        "icd10": "K52.9",
        "symptoms": ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"],
    },
    "Bronchial Asthma": {
        "icd10": "J45.909",
        "symptoms": ["fatigue", "cough", "high_fever", "breathlessness", "family_history", "mucoid_sputum"],
    },
    "Hypertension": {
        "icd10": "I10",
        "symptoms": ["headache", "chest_pain", "dizziness", "loss_of_balance", "lack_of_concentration"],
    },
    "Migraine": {
        "icd10": "G43.909",
        "symptoms": ["acidity", "indigestion", "headache", "blurred_and_distorted_vision", "excessive_hunger", "stiff_neck", "depression", "irritability", "visual_disturbances"],
    },
    "Cervical spondylosis": {
        "icd10": "M47.812",
        "symptoms": ["back_pain", "weakness_in_limbs", "neck_pain", "dizziness", "loss_of_balance"],
    },
    "Paralysis (brain hemorrhage)": {
        "icd10": "I61.9",
        "symptoms": ["vomiting", "headache", "weakness_of_one_body_side", "altered_sensorium"],
    },
    "Jaundice": {
        "icd10": "R17",
        "symptoms": ["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "yellowish_skin", "dark_urine", "abdominal_pain"],
    },
    "Malaria": {
        "icd10": "B54",
        "symptoms": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "diarrhoea", "muscle_pain"],
    },
    "Chicken pox": {
        "icd10": "B01.9",
        "symptoms": ["itching", "skin_rash", "fatigue", "lethargy", "high_fever", "headache", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "malaise", "red_spots_over_body"],
    },
    "Dengue": {
        "icd10": "A90",
        "symptoms": ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "muscle_pain", "red_spots_over_body"],
    },
    "Typhoid": {
        "icd10": "A01.00",
        "symptoms": ["chills", "vomiting", "fatigue", "high_fever", "nausea", "constipation", "abdominal_pain", "diarrhoea", "toxic_look_typhos", "belly_pain"],
    },
    "Hepatitis A": {
        "icd10": "B15.9",
        "symptoms": ["joint_pain", "vomiting", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "diarrhoea", "mild_fever", "yellowing_of_eyes", "muscle_pain"],
    },
    "Hepatitis B": {
        "icd10": "B16.9",
        "symptoms": ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite", "abdominal_pain", "yellow_urine", "yellowing_of_eyes", "malaise", "receiving_blood_transfusion", "receiving_unsterile_injections"],
    },
    "Hepatitis C": {
        "icd10": "B17.10",
        "symptoms": ["fatigue", "yellowish_skin", "nausea", "loss_of_appetite", "family_history"],
    },
    "Hepatitis D": {
        "icd10": "B17.0",
        "symptoms": ["joint_pain", "vomiting", "fatigue", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes"],
    },
    "Hepatitis E": {
        "icd10": "B17.2",
        "symptoms": ["joint_pain", "vomiting", "fatigue", "high_fever", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes", "acute_liver_failure", "stomach_bleeding"],
    },
    "Alcoholic hepatitis": {
        "icd10": "K70.10",
        "symptoms": ["vomiting", "yellowish_skin", "abdominal_pain", "swelling_of_stomach", "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload"],
    },
    "Tuberculosis": {
        "icd10": "A15.9",
        "symptoms": ["chills", "vomiting", "fatigue", "weight_loss", "cough", "high_fever", "breathlessness", "sweating", "loss_of_appetite", "mild_fever", "yellowing_of_eyes", "swelled_lymph_nodes", "malaise", "phlegm", "chest_pain", "blood_in_sputum"],
    },
    "Common Cold": {
        "icd10": "J00",
        "symptoms": ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever", "headache", "swelled_lymph_nodes", "malaise", "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain", "loss_of_smell", "muscle_pain"],
    },
    "Pneumonia": {
        "icd10": "J18.9",
        "symptoms": ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "malaise", "phlegm", "chest_pain", "fast_heart_rate", "rusty_sputum"],
    },
    "Dimorphic hemmorhoids(piles)": {
        "icd10": "K64.9",
        "symptoms": ["constipation", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus"],
    },
    "Heart attack": {
        "icd10": "I21.9",
        "symptoms": ["vomiting", "breathlessness", "sweating", "chest_pain"],
    },
    "Varicose veins": {
        "icd10": "I83.90",
        "symptoms": ["fatigue", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "prominent_veins_on_calf"],
    },
    "Hypothyroidism": {
        "icd10": "E03.9",
        "symptoms": ["fatigue", "weight_gain", "cold_hands_and_feets", "mood_swings", "lethargy", "dizziness", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "depression", "irritability", "abnormal_menstruation"],
    },
    "Hyperthyroidism": {
        "icd10": "E05.90",
        "symptoms": ["fatigue", "mood_swings", "weight_loss", "restlessness", "sweating", "diarrhoea", "fast_heart_rate", "excessive_hunger", "muscle_weakness", "irritability", "abnormal_menstruation"],
    },
    "Hypoglycemia": {
        "icd10": "E16.2",
        "symptoms": ["vomiting", "fatigue", "anxiety", "sweating", "headache", "nausea", "blurred_and_distorted_vision", "excessive_hunger", "slurred_speech", "irritability", "palpitations", "drying_and_tingling_lips"],
    },
    "Osteoarthristis": {
        "icd10": "M19.90",
        "symptoms": ["joint_pain", "neck_pain", "knee_pain", "hip_joint_pain", "swelling_joints", "painful_walking"],
    },
    "Arthritis": {
        "icd10": "M19.9",
        "symptoms": ["muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "painful_walking"],
    },
    "(vertigo) Paroymsal Positional Vertigo": {
        "icd10": "H81.10",
        "symptoms": ["vomiting", "headache", "nausea", "spinning_movements", "loss_of_balance", "unsteadiness"],
    },
    "Acne": {
        "icd10": "L70.0",
        "symptoms": ["skin_rash", "pus_filled_pimples", "blackheads", "scurring"],
    },
    "Urinary tract infection": {
        "icd10": "N39.0",
        "symptoms": ["burning_micturition", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine"],
    },
    "Psoriasis": {
        "icd10": "L40.9",
        "symptoms": ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails"],
    },
    "Impetigo": {
        "icd10": "L01.0",
        "symptoms": ["skin_rash", "high_fever", "blister", "red_sore_around_nose", "yellow_crust_ooze"],
    },
}


def _build_feature_space() -> list[str]:
    """Collect every symptom mentioned across all diseases as a sorted feature list."""
    seen: set[str] = set()
    for d in DISEASES.values():
        seen.update(d["symptoms"])

    distractors = [
        "abdominal_pain", "anxiety", "back_pain", "chest_pain", "chills", "cough", "diarrhoea",
        "dizziness", "fatigue", "fever", "headache", "high_fever", "joint_pain", "loss_of_appetite",
        "muscle_pain", "nausea", "rash", "sore_throat", "sweating", "vomiting", "weight_loss",
        "weight_gain", "blurred_vision", "constipation", "dehydration", "shortness_of_breath",
    ]
    seen.update(distractors)
    return sorted(seen)


def _generate_dataset(rng: random.Random) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    feature_names = _build_feature_space()
    feat_index = {name: i for i, name in enumerate(feature_names)}

    diseases_sorted = sorted(DISEASES.keys())
    label_to_id = {d: i for i, d in enumerate(diseases_sorted)}

    rows: list[np.ndarray] = []
    labels: list[int] = []

    for disease, info in DISEASES.items():
        typical = info["symptoms"]
        not_typical = [s for s in feature_names if s not in typical]
        for _ in range(N_SAMPLES_PER_DISEASE):
            row = np.zeros(len(feature_names), dtype=np.float32)

            for s in typical:
                if rng.random() > NOISE_PROB_OFF:
                    row[feat_index[s]] = 1.0

            for s in not_typical:
                if rng.random() < NOISE_PROB_ON:
                    row[feat_index[s]] = 1.0
            rows.append(row)
            labels.append(label_to_id[disease])

    x = np.vstack(rows)
    y = np.array(labels, dtype=np.int32)
    return x, y, feature_names, label_to_id


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RNG_SEED)
    np.random.seed(RNG_SEED)

    print(f"Building dataset: {len(DISEASES)} diseases x {N_SAMPLES_PER_DISEASE} samples = {len(DISEASES) * N_SAMPLES_PER_DISEASE} rows")
    x, y, feature_names, label_to_id = _generate_dataset(rng)
    print(f"Feature space: {len(feature_names)} symptoms")
    print(f"Class space:   {len(label_to_id)} diseases")


    perm = np.random.permutation(len(x))
    x, y = x[perm], y[perm]

    print("Training LightGBM multiclass classifier...")
    train_data = lgb.Dataset(x, label=y, feature_name=feature_names)
    params = {
        "objective": "multiclass",
        "num_class": len(label_to_id),
        "metric": "multi_logloss",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": RNG_SEED,
    }
    model = lgb.train(params, train_data, num_boost_round=120)


    train_pred = np.asarray(model.predict(x)).argmax(axis=1)
    acc = float(np.mean(train_pred == y))
    print(f"Training accuracy: {acc:.3f}")


    model_path = OUTPUT_DIR / "model.txt"
    model.save_model(str(model_path))

    feature_names_path = OUTPUT_DIR / "feature_names.json"
    feature_names_path.write_text(json.dumps(feature_names, indent=2))


    label_map: dict[str, dict] = {}
    for disease, idx in label_to_id.items():
        icd10 = DISEASES[disease]["icd10"]
        label_map[icd10] = {"id": idx, "name": disease}
    label_map_path = OUTPUT_DIR / "label_map.json"
    label_map_path.write_text(json.dumps(label_map, indent=2))

    print(f"Saved model      -> {model_path}")
    print(f"Saved features   -> {feature_names_path}")
    print(f"Saved label_map  -> {label_map_path}")


if __name__ == "__main__":
    main()
