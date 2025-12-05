import argparse
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump

"""
XGBOOST TRAINING FOR FANTASY PLAYER PERFORMANCE
------------------------------------------------

Input:
- TDA features (.npy)
- Optional target CSV for player scores (fantasy points)

Output:
- Trained XGBoost model saved to disk
- Optional evaluation metrics
"""


def train_xgboost_model(X, y, save_dir):
    print("Training XGBoost regressor...")

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=20,
        verbose=True,
    )

    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"✔ Validation RMSE: {rmse:.4f}")

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "xgboost_model.joblib")
    dump(model, model_path)
    print(f"✔ XGBoost model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=str, required=True, help="Path to TDA features (.npy)"
    )
    parser.add_argument(
        "--target_csv", type=str, default=None, help="CSV with target player points"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save trained model"
    )
    args = parser.parse_args()

    # Load TDA features
    X = np.load(args.features)

    # Load target values
    if args.target_csv and os.path.exists(args.target_csv):
        df_target = pd.read_csv(args.target_csv)
        # Assumes target column is 'fantasy_points'
        if "fantasy_points" not in df_target.columns:
            raise ValueError("CSV must contain 'fantasy_points' column")
        y = df_target["fantasy_points"].values
    else:
        print("⚠ No target CSV provided, using random dummy target values")
        y = np.random.rand(X.shape[0])

    train_xgboost_model(X, y, args.save_dir)
