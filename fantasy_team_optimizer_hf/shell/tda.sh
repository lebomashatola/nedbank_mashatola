#!/bin/bash
set -e

INPUT_CSV="data/players.csv"
TDA_OUTPUT="data/tda_features.npy"
CORR_OUTPUT="data/feature_correlation.npy"

python3 src/tda_encoder.py --input_csv $INPUT_CSV --tda_output $TDA_OUTPUT --corr_output $CORR_OUTPUT

echo "TDA encoding and distance correlation complete!"
