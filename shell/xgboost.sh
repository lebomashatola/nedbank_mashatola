#!/bin/bash
set -e

TDA_FEATURES="data/tda_features.npy"
TARGET_CSV="data/players_targets.csv"
MODEL_DIR="models"

mkdir -p $MODEL_DIR

python3 src/xgboost_training.py --features $TDA_FEATURES --target_csv $TARGET_CSV --save_dir $MODEL_DIR

echo "✔ XGBoost training complete!"
