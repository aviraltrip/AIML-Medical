import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd


def generate_synthetic_data(num_samples=5000):
    np.random.seed(42)


    age = np.random.randint(18, 80, size=num_samples)
    distance_to_phc = np.random.uniform(0.5, 30.0, size=num_samples)


    occ_rand = np.random.rand(num_samples)
    occupation_farming = (occ_rand < 0.35).astype(float)
    occupation_laborer = ((occ_rand >= 0.35) & (occ_rand < 0.75)).astype(float)

    tobacco_alcohol_usage = (np.random.rand(num_samples) < 0.40).astype(float)
    previous_adherence = (np.random.rand(num_samples) < 0.70).astype(float)


    severity_tier_val = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.3, 0.4, 0.15, 0.1, 0.05])





    z = (
        -1.2
        + 0.12 * distance_to_phc
        + 0.65 * occupation_farming
        + 0.80 * occupation_laborer
        + 0.35 * tobacco_alcohol_usage
        - 0.55 * severity_tier_val
        - 1.50 * previous_adherence
        + 0.008 * (age - 45)
    )


    prob_default = 1.0 / (1.0 + np.exp(-z))


    loss_to_follow_up = (np.random.rand(num_samples) < prob_default).astype(int)

    df = pd.DataFrame({
        "age": age,
        "distance_to_phc": distance_to_phc,
        "occupation_farming": occupation_farming,
        "occupation_laborer": occupation_laborer,
        "tobacco_alcohol_usage": tobacco_alcohol_usage,
        "severity_tier_val": severity_tier_val,
        "previous_adherence": previous_adherence,
        "loss_to_follow_up": loss_to_follow_up
    })

    return df

def main():
    print("Generating synthetic rural adherence data...")
    df = generate_synthetic_data(5000)

    features = [
        "age",
        "distance_to_phc",
        "occupation_farming",
        "occupation_laborer",
        "tobacco_alcohol_usage",
        "severity_tier_val",
        "previous_adherence"
    ]
    target = "loss_to_follow_up"

    x = df[features]
    y = df[target]

    print(f"Dataset defaults baseline rate: {y.mean() * 100:.2f}%")


    train_data = lgb.Dataset(x, label=y)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "seed": 42,
        "verbose": -1
    }

    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100
    )


    model_dir = "models/adherence_lgbm"
    os.makedirs(model_dir, exist_ok=True)


    model.save_model(os.path.join(model_dir, "model.txt"))

    with open(os.path.join(model_dir, "feature_names.json"), "w") as f:
        json.dump(features, f)

    label_map = {
        "adhered": 0,
        "loss_to_follow_up": 1
    }
    with open(os.path.join(model_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f)

    print("Model trained and artifacts saved to models/adherence_lgbm/ successfully!")

if __name__ == "__main__":
    main()
