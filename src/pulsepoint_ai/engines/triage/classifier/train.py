"""LightGBM triage classifier trainer. CLI entrypoint:

    uv run python -m pulsepoint_ai.engines.triage.classifier.train \
        --data data/processed/triage_train.parquet \
        --out models/triage_lgbm

Outputs: model.txt, feature_names.json, label_map.json, metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from pulsepoint_ai.engines.triage.classifier.features import feature_names

SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH", "URGENT", "EMERGENCY"]


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    df = pd.read_parquet(path)
    label_map = {label: i for i, label in enumerate(SEVERITY_LABELS)}
    y = df["severity"].map(label_map).to_numpy()
    feat_cols = [c for c in df.columns if c != "severity"]
    x = df[feat_cols].to_numpy(dtype=np.float32)
    return x, y


def train(
    x: np.ndarray,  
    y: np.ndarray,  
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[lgb.Booster, dict[str, Any]]:
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
        "seed": seed,
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_f1s: list[float] = []
    for _, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        d_train = lgb.Dataset(x[train_idx], label=y[train_idx])
        d_val = lgb.Dataset(x[val_idx], label=y[val_idx], reference=d_train)
        booster = lgb.train(
            params,
            d_train,
            num_boost_round=500,
            valid_sets=[d_val],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        preds = np.asarray(booster.predict(x[val_idx])).argmax(axis=1)
        f1 = f1_score(y[val_idx], preds, average="macro")
        fold_f1s.append(f1)


    d_full = lgb.Dataset(x, label=y)
    final = lgb.train(params, d_full, num_boost_round=500)

    metrics = {
        "cv_macro_f1_mean": float(np.mean(fold_f1s)),
        "cv_macro_f1_std": float(np.std(fold_f1s)),
        "n_samples": len(y),
        "n_features": x.shape[1],
        "label_distribution": {
            SEVERITY_LABELS[i]: int((y == i).sum()) for i in range(len(SEVERITY_LABELS))
        },
        "report": classification_report(
            y, np.asarray(final.predict(x)).argmax(axis=1), target_names=SEVERITY_LABELS, output_dict=True
        ),
    }
    return final, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    x, y = load_dataset(args.data)
    booster, metrics = train(x, y)

    args.out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.out / "model.txt"))
    (args.out / "feature_names.json").write_text(json.dumps(feature_names(), indent=2))
    (args.out / "label_map.json").write_text(
        json.dumps({label: i for i, label in enumerate(SEVERITY_LABELS)}, indent=2)
    )
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Trained. Macro-F1 = {metrics['cv_macro_f1_mean']:.3f}. Saved to {args.out}.")


if __name__ == "__main__":
    main()
