"""Train the LightGBM triage severity classifier (5 tiers).

Mirrors the disease-classifier pattern: a synthetic dataset is generated locally
from clinical rules (vital-sign ranges per tier and cardinal symptom clusters),
then a LightGBM multiclass model is fit. SHAP analysis is run on the validation
slice and saved alongside the model artifacts for explainability.

The feature space matches what `engines/triage/classifier/infer.py` expects:
    - 130 canonical symptom names from configs/symptoms.yaml (one-hot, snake_case)
    - 7 continuous vitals using infer.py's primary alias names

Outputs (all written to models/triage_lgbm/):
    model.txt              - LightGBM Booster
    feature_names.json     - ordered feature list (137 entries)
    label_map.json         - {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "URGENT": 3, "EMERGENCY": 4}
    shap_importance.json   - top-20 SHAP-ranked features (mean |SHAP|)
    metrics.json           - train/val accuracy + per-tier breakdown

Run:
    python scripts/train_triage_classifier.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import yaml

OUTPUT_DIR = Path("models/triage_lgbm")
RNG_SEED = 42
SAMPLES_PER_TIER = 1000
SYMPTOM_DROP_PROB = 0.05
DISTRACTOR_PROB = 0.01

SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH", "URGENT", "EMERGENCY"]


VITAL_FEATURES = [
    "spo2", "bp_systolic", "bp_diastolic", "blood_sugar_mg_dl",
    "pulse_bpm", "temp_c", "respiratory_rate",
]


VITAL_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "LOW":       {"spo2": (96, 100), "bp_systolic": (110, 130), "bp_diastolic": (70, 85),
                  "blood_sugar_mg_dl": (80, 110), "pulse_bpm": (60, 90),
                  "temp_c": (36.4, 37.2), "respiratory_rate": (12, 18)},
    "MEDIUM":    {"spo2": (94, 98), "bp_systolic": (115, 145), "bp_diastolic": (75, 92),
                  "blood_sugar_mg_dl": (75, 160), "pulse_bpm": (75, 110),
                  "temp_c": (37.5, 38.9), "respiratory_rate": (14, 22)},
    "HIGH":      {"spo2": (92, 96), "bp_systolic": (105, 165), "bp_diastolic": (70, 100),
                  "blood_sugar_mg_dl": (60, 220), "pulse_bpm": (90, 130),
                  "temp_c": (36.5, 39.5), "respiratory_rate": (16, 26)},
    "URGENT":    {"spo2": (90, 94), "bp_systolic": (85, 200), "bp_diastolic": (55, 118),
                  "blood_sugar_mg_dl": (55, 350), "pulse_bpm": (110, 150),
                  "temp_c": (35.0, 40.5), "respiratory_rate": (22, 32)},
    "EMERGENCY": {"spo2": (75, 91), "bp_systolic": (60, 90), "bp_diastolic": (35, 60),
                  "blood_sugar_mg_dl": (25, 500), "pulse_bpm": (35, 200),
                  "temp_c": (33.5, 41.5), "respiratory_rate": (4, 40)},
}


SYMPTOMS_PER_TIER: dict[str, list[list[str]]] = {
    "LOW": [
        ["runny_nose", "sneezing", "sore_throat"],
        ["cough_dry"],
        ["headache"],
        ["fatigue"],
        ["nasal_congestion", "sneezing"],
        ["heartburn"],
        ["bloating"],
        ["itching"],
        ["constipation"],
        ["morning_stiffness"],
    ],
    "MEDIUM": [
        ["fever", "fatigue"],
        ["abdominal_pain", "nausea"],
        ["vomiting", "diarrhea"],
        ["headache", "nausea"],
        ["back_pain"],
        ["joint_pain", "fatigue"],
        ["urinary_frequency", "dysuria"],
        ["rash"],
        ["dehydration"],
        ["ear_pain", "fever"],
    ],
    "HIGH": [
        ["chest_pain"],
        ["shortness_of_breath", "wheezing"],
        ["severe_headache_sudden"],
        ["abdominal_pain_severe", "vomiting"],
        ["fever", "neck_stiffness"],
        ["high_fever", "rigors"],
        ["palpitations", "chest_tightness"],
        ["hematuria", "flank_pain"],
        ["jaundice", "abdominal_pain"],
        ["confusion"],
    ],
    "URGENT": [
        ["chest_pain", "shortness_of_breath"],
        ["abdominal_pain_severe", "rebound_tenderness"],
        ["high_fever", "confusion"],
        ["hemoptysis"],
        ["head_injury"],
        ["vomiting_blood"],
        ["bloody_stool"],
        ["seizure"],
        ["severe_headache_sudden", "vomiting"],
        ["dehydration", "lethargy_infant"],
    ],
    "EMERGENCY": [
        ["chest_pain", "shortness_of_breath", "cold_sweats", "sweating_excessive"],
        ["facial_droop", "slurred_speech", "weakness_one_side"],
        ["loss_of_consciousness"],
        ["anaphylaxis_signs", "throat_swelling", "tongue_swelling"],
        ["seizure", "loss_of_consciousness"],
        ["severe_bleeding", "loss_of_consciousness"],
        ["chest_pain", "syncope"],
        ["cyanosis", "shortness_of_breath"],
        ["severe_headache_sudden", "neck_stiffness", "photophobia"],
        ["rigid_abdomen", "rebound_tenderness", "high_fever"],
    ],
}


def _load_symptom_names() -> list[str]:
    with open("configs/symptoms.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [s["name"] for s in cfg["symptoms"]]


def _generate_dataset(rng: random.Random) -> tuple[np.ndarray, np.ndarray, list[str]]:
    symptom_names = _load_symptom_names()
    feature_names = symptom_names + VITAL_FEATURES
    feat_idx = {n: i for i, n in enumerate(feature_names)}

    rows: list[np.ndarray] = []
    labels: list[int] = []

    for tier_id, tier_name in enumerate(SEVERITY_LABELS):
        patterns = SYMPTOMS_PER_TIER[tier_name]
        vital_range = VITAL_RANGES[tier_name]

        for _ in range(SAMPLES_PER_TIER):
            row = np.zeros(len(feature_names), dtype=np.float32)

            pattern = rng.choice(patterns)
            for s in pattern:
                if s in feat_idx and rng.random() > SYMPTOM_DROP_PROB:
                    row[feat_idx[s]] = 1.0

            for s in symptom_names:
                if s not in pattern and rng.random() < DISTRACTOR_PROB:
                    row[feat_idx[s]] = 1.0

            for v_name, (lo, hi) in vital_range.items():
                row[feat_idx[v_name]] = rng.uniform(lo, hi)

            rows.append(row)
            labels.append(tier_id)

    return np.vstack(rows), np.array(labels, dtype=np.int32), feature_names


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RNG_SEED)
    np.random.seed(RNG_SEED)

    print(f"Generating dataset: {len(SEVERITY_LABELS)} tiers x {SAMPLES_PER_TIER} samples = {len(SEVERITY_LABELS) * SAMPLES_PER_TIER} rows")
    x, y, feature_names = _generate_dataset(rng)
    print(f"Feature space: {len(feature_names)} ({len(feature_names) - len(VITAL_FEATURES)} symptoms + {len(VITAL_FEATURES)} vitals)")

    perm = np.random.permutation(len(x))
    x, y = x[perm], y[perm]
    split = int(len(x) * 0.8)
    x_train, x_val, y_train, y_val = x[:split], x[split:], y[:split], y[split:]

    print(f"Training LightGBM ({len(x_train)} train / {len(x_val)} val)...")
    train_data = lgb.Dataset(x_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(x_val, label=y_val, feature_name=feature_names, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": len(SEVERITY_LABELS),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": RNG_SEED,
    }
    model = lgb.train(
        params, train_data, num_boost_round=300,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    val_preds = np.argmax(model.predict(x_val), axis=1)
    train_preds = np.argmax(model.predict(x_train), axis=1)
    train_acc = float(np.mean(train_preds == y_train))
    val_acc = float(np.mean(val_preds == y_val))
    print(f"\nTrain accuracy: {train_acc:.3f}")
    print(f"Val accuracy:   {val_acc:.3f}")

    per_tier: dict[str, dict] = {}
    print("Per-tier validation accuracy:")
    for i, tier in enumerate(SEVERITY_LABELS):
        mask = y_val == i
        if mask.sum() > 0:
            tier_acc = float(np.mean(val_preds[mask] == y_val[mask]))
            per_tier[tier] = {"accuracy": tier_acc, "n_samples": int(mask.sum())}
            print(f"  {tier:10}: {tier_acc:.3f} ({int(mask.sum())} samples)")


    shap_importance: list[dict] = []
    try:
        import shap
        print("\nComputing SHAP feature importance (TreeExplainer)...")
        explainer = shap.TreeExplainer(model)
        sample = x_val[:300]
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            stacked = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])
            mean_abs = stacked.mean(axis=0)
        else:
            arr = np.abs(shap_values)
            while arr.ndim > 1:
                arr = arr.mean(axis=0)
            mean_abs = arr
        top_idx = np.argsort(mean_abs)[::-1][:20]
        shap_importance = [
            {"feature": feature_names[int(i)], "mean_abs_shap": float(mean_abs[int(i)])}
            for i in top_idx
        ]
        print(f"Top 5 SHAP features: {[f['feature'] for f in shap_importance[:5]]}")
    except Exception as e:
        print(f"SHAP analysis skipped: {e}")


    model.save_model(str(OUTPUT_DIR / "model.txt"))
    (OUTPUT_DIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (OUTPUT_DIR / "label_map.json").write_text(
        json.dumps({label: i for i, label in enumerate(SEVERITY_LABELS)}, indent=2)
    )
    if shap_importance:
        (OUTPUT_DIR / "shap_importance.json").write_text(json.dumps(shap_importance, indent=2))

    metrics = {
        "n_samples": len(x),
        "n_features": len(feature_names),
        "n_classes": len(SEVERITY_LABELS),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "best_iteration": int(model.best_iteration or 0),
        "per_tier_accuracy": per_tier,
        "seed": RNG_SEED,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nArtifacts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
