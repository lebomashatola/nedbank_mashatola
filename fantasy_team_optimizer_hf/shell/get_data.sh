#!/bin/bash
set -e

echo "Starting data acquisition..."

CONFIG_FILE="configs/config.yaml"
DATA_DIR="data"

mkdir -p $DATA_DIR

# Read dataset source from config
DATASET=$(grep "^dataset:" $CONFIG_FILE | awk '{print $2}')

echo "Using dataset source: $DATASET"

if [ "$DATASET" = "espn" ]; then
    echo "Fetching ESPN public API data..."
    python3 src/data_loader.py --source espn --output $DATA_DIR/players_raw.json

elif [ "$DATASET" = "thesportsdb" ]; then
    echo "Fetching TheSportsDB API data..."
    python3 src/data_loader.py --source thesportsdb --output $DATA_DIR/players_raw.json

else
    echo "Using local CSV data..."
    python3 src/data_loader.py --source local --output $DATA_DIR/players_raw.json
fi

echo "Data acquisition complete!"
