import argparse
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from joblib import dump

# HuggingFace imports
from transformers import TabularConfig, TabularModel, TrainingArguments, Trainer
from torch.utils.data import Dataset

"""
MODEL TRAINING
--------------
Input: TDA features (.npy)
Output: Trained ML model saved to disk
Supports:
- HuggingFace Tabular Transformer (Apache 2.0)
- XGBoost
- RandomForest
"""


# -------------------------
# PyTorch Dataset
# -------------------------
class PlayerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"input_features": self.X[idx], "labels": self.y[idx]}


# -------------------------
# HF Tabular Transformer Trainer
# -------------------------
def train_hf_transformer(X, y, save_dir):
    print("🤖 Training HuggingFace Tabular Transformer...")

    train_idx, val_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42
    )
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_dataset = PlayerDataset(X_train, y_train)
    val_dataset = PlayerDataset(X_val, y_val)

    # Tabular config
    config = TabularConfig(
        num_features=X.shape[1], num_labels=1, hidden_sizes=[128, 64], dropout=0.1
    )

    model = TabularModel(config)

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-3,
        logging_dir=os.path.join(save_dir, "logs"),
        logging_steps=10,
        save_total_limit=1,
        fp16=False,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        mse = ((preds.flatten() - labels.flatten()) ** 2).mean()
        return {"mse": mse}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(save_dir)
    print(f"✔ HuggingFace model saved to {save_dir}")


# -------------------------
# XGBoost Trainer
# -------------------------
def train_xgboost(X, y, save_dir):
    print("🤖 Training XGBoost regressor...")
    model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05)
    model.fit(X, y)
    os.makedirs(save_dir, exist_ok=True)
    dump(model, os.path.join(save_dir, "xgboost_model.joblib"))
    print(f"✔ XGBoost model saved to {save_dir}")


# -------------------------
# RandomForest Trainer
# -------------------------
def train_random_forest(X, y, save_dir):
    print("🤖 Training RandomForest regressor...")
    model = RandomForestRegressor(n_estimators=300, max_depth=10, n_jobs=-1)
    model.fit(X, y)
    os.makedirs(save_dir, exist_ok=True)
    dump(model, os.path.join(save_dir, "rf_model.joblib"))
    print(f"✔ RandomForest model saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["huggingface", "xgboost", "randomforest"],
    )
    parser.add_argument(
        "--features", type=str, required=True, help="Path to .npy TDA features"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save trained model"
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    X = np.load(args.features)
    # Dummy target for now; replace with actual player fantasy points
    # For real usage, load target from CSV or API
    y = np.random.rand(X.shape[0])

    if args.model == "huggingface":
        train_hf_transformer(X, y, args.save_dir)
    elif args.model == "xgboost":
        train_xgboost(X, y, args.save_dir)
    else:
        train_random_forest(X, y, args.save_dir)
