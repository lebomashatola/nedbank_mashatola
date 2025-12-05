#!/bin/bash
set -e

FEATURES="data/tda_features.npy"
METADATA="data/players_metadata.csv"
OUTPUT="output/team.json"
BUDGET=100
MAX_PER_TEAM=3

python3 src/optimizer.py --features $FEATURES --metadata $METADATA --output $OUTPUT \
    --budget $BUDGET --max_per_team $MAX_PER_TEAM

echo "✔ Optimization complete! Team saved to $OUTPUT"
