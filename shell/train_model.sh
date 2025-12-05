#!/bin/bash
set -e

CONFIG="configs/config.yaml"
TDA_FEATURES="data/tda_features.npy"
MODEL_DIR="models"

mkdir -p $MODEL_DIR

MODEL_TYPE=$(grep "^model:" $CONFIG | awk '{print $2}')
echo "Selected model: $MODEL_TYPE"

python3 src/model_training.py --model $MODEL_TYPE --features $TDA_FEATURES --save_dir $MODEL_DIR

echo "✔ Model training complete!"
