"""LightGBM disease classifier trainer. CLI:

    uv run python -m pulsepoint_ai.engines.predict.disease_classifier.train \
        --data data/processed/disease_train.parquet --out models/disease_lgbm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import train_test_split

from pulsepoint_ai.engines.triage.classifier.features import feature_names


def load_dataset(path: Path) -> tuple[NDArray[np.float32], NDArray[np.int64], list[str]]:
    df = pd.read_parquet(path)
    labels = [str(lbl) for lbl in sorted(df["icd10"].unique())]
    label_map = {label: i for i, label in enumerate(labels)}
    y = df["icd10"].map(label_map).to_numpy(dtype=np.int64)
    feat_cols = [c for c in df.columns if c not in {"icd10", "name"}]
    x = df[feat_cols].to_numpy(dtype=np.float32)
    return x, y, labels


def train_model(
    x: NDArray[np.float32],
    y: NDArray[np.int64],
    n_classes: int,
) -> lgb.Booster:
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    d_train = lgb.Dataset(x_tr, label=y_tr)
    d_val = lgb.Dataset(x_val, label=y_val, reference=d_train)
    booster = lgb.train(
        params,
        d_train,
        num_boost_round=800,
        valid_sets=[d_val],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
    )
    return booster


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    x, y, labels = load_dataset(args.data)
    booster = train_model(x, y, len(labels))

    proba = np.asarray(booster.predict(x))
    metrics = {
        "top1_acc": float(top_k_accuracy_score(y, proba, k=1, labels=range(len(labels)))),
        "top5_acc": float(top_k_accuracy_score(y, proba, k=5, labels=range(len(labels)))),
        "n_samples": len(y),
        "n_classes": len(labels),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.out / "model.txt"))
    (args.out / "feature_names.json").write_text(json.dumps(feature_names(), indent=2))
    (args.out / "label_map.json").write_text(
        json.dumps({label: i for i, label in enumerate(labels)}, indent=2)
    )
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Trained. top1={metrics['top1_acc']:.3f} top5={metrics['top5_acc']:.3f}")


if __name__ == "__main__":
    main()
