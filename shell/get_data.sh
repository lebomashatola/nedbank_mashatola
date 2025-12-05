#!/bin/bash
set -e

echo "Starting data acquisition"

#CONFIG_FILE="configs/config.yaml"
CONFIG_FILE="/Users/lebohangmashatola/documents/nedbank_mashatola/fantasy_team_optimizer_hf/configs/config.yaml"
DATA_DIR="data"

mkdir -p $DATA_DIR

# Read dataset source from config
DATASET=$(grep "^dataset:" $CONFIG_FILE | awk '{print $2}')
    
if [ "$DATASET" = "espn" ]; then
    echo "Fetching ESPN public API data..."
    python3 src/data_loader.py --source espn --output $DATA_DIR/players_raw.json

elif [ "$DATASET" = "thesportsdb" ]; then
    echo "Fetching TheSportsDB API data..."
    python3 src/data_loader.py --source thesportsdb --output $DATA_DIR/players_raw.json

elif [ "$DATASET" = "synthetic" ]; then
    echo "Fetching Synthetic data..."
    python3 src/data_loader.py --source synthetic --output $DATA_DIR/players_raw.json

fi

echo "Data acquisition complete!"
