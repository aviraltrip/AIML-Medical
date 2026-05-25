import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

def generate_synthetic_data(num_samples=5000):
    np.random.seed(42)
    
    # 1. Generate features
    age = np.random.randint(18, 80, size=num_samples)
    distance_to_phc = np.random.uniform(0.5, 30.0, size=num_samples)
    
    # Occupation categories (Farming: 35%, Manual Laborer: 40%, Other: 25%)
    occ_rand = np.random.rand(num_samples)
    occupation_farming = (occ_rand < 0.35).astype(float)
    occupation_laborer = ((occ_rand >= 0.35) & (occ_rand < 0.75)).astype(float)
    
    tobacco_alcohol_usage = (np.random.rand(num_samples) < 0.40).astype(float)
    previous_adherence = (np.random.rand(num_samples) < 0.70).astype(float)
    
    # Severity tier (0: LOW, 1: MEDIUM, 2: HIGH, 3: URGENT, 4: EMERGENCY)
    severity_tier_val = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.3, 0.4, 0.15, 0.1, 0.05])
    
    # 2. Define probability of non-adherence (loss-to-follow-up) using a logit model
    # Base constant + weights
    # positive weight = increases default probability (loss to follow up)
    # negative weight = increases adherence (decreases default probability)
    z = (
        -1.2
        + 0.12 * distance_to_phc              # Farther distance -> higher default
        + 0.65 * occupation_farming          # Farming chores -> less time to travel -> higher default
        + 0.80 * occupation_laborer          # Daily wage loss -> higher default
        + 0.35 * tobacco_alcohol_usage        # Unhealthy habits -> higher default
        - 0.55 * severity_tier_val            # High severity -> patient/ASHA takes it seriously -> lower default
        - 1.50 * previous_adherence          # High adherence history -> lower default
        + 0.008 * (age - 45)                 # Slightly higher default for elderly
    )
    
    # Logit formula
    prob_default = 1.0 / (1.0 + np.exp(-z))
    
    # Add random noise
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
    
    X = df[features]
    y = df[target]
    
    print(f"Dataset defaults baseline rate: {y.mean() * 100:.2f}%")
    
    # Train LightGBM Model
    train_data = lgb.Dataset(X, label=y)
    
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
    
    # Create target directory
    model_dir = "models/adherence_lgbm"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save files
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
