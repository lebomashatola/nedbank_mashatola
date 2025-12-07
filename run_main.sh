#!/bin/bash
# run_main.sh
# Bash script to run the main.py script for Fantasy Football Player Performance System

# Activate conda environment
echo "Activating conda environment..."
conda activate fantasy_football_env

# Run main.py
echo "Running main.py..."
python main.py

# Optional: deactivate environment after run
echo "Deactivating conda environment..."
conda deactivate
