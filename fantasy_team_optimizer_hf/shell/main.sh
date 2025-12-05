#!/bin/bash
set -e

echo "🚀 Running full Fantasy Team Optimizer pipeline..."

sh scripts/obtain_data.sh
sh scripts/run_tda.sh
sh scripts/train_model.sh
sh scripts/run_optimizer.sh

echo "Pipeline complete!"
echo "Final team available at: output/team.json"
