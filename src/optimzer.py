import argparse
import os
import json
import numpy as np
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary

"""
FANTASY TEAM OPTIMIZER
----------------------

Input:
- Model predictions (.npy)
- Player metadata CSV (positions, salaries, team)
- Config constraints (budget, max players per position/team)

Output:
- JSON file with selected team lineup
"""


# -------------------------
# Load data
# -------------------------
def load_data(predictions_file, metadata_file):
    y_pred = np.load(predictions_file)  # predicted scores
    df_meta = pd.read_csv(metadata_file)
    if len(y_pred) != len(df_meta):
        raise ValueError("Number of predictions does not match number of players")
    df_meta["predicted_score"] = y_pred
    return df_meta


# -------------------------
# Integer Programming Optimizer
# -------------------------
def optimize_team(df, budget=100, positions=None, max_per_team=3, squad_size=11):
    """
    df: DataFrame with player metadata + predicted_score
    positions: dict like {"GK":1, "DEF":4, "MID":4, "FWD":2}
    """
    if positions is None:
        positions = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}

    n = len(df)
    prob = LpProblem("Fantasy_Team_Optimization", LpMaximize)

    # Decision variables: 1 if player selected, 0 otherwise
    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]

    # Objective: maximize predicted score
    prob += lpSum(x[i] * df.loc[i, "predicted_score"] for i in range(n))

    # Constraint: total squad size
    prob += lpSum(x) == squad_size

    # Constraint: budget
    prob += lpSum(x[i] * df.loc[i, "salary"] for i in range(n)) <= budget

    # Constraint: position limits
    for pos, count in positions.items():
        prob += lpSum(x[i] for i in range(n) if df.loc[i, "position"] == pos) == count

    # Constraint: max players per real team
    teams = df["team"].unique()
    for team in teams:
        prob += (
            lpSum(x[i] for i in range(n) if df.loc[i, "team"] == team) <= max_per_team
        )

    # Solve
    prob.solve()

    # Extract selected players
    selected = df[[x[i].value() == 1 for i in range(n)]].copy()
    return selected


# -------------------------
# Main
# -------------------------
def run_optimizer(predictions_file, metadata_file, output_file, budget, max_per_team):
    print("⚡ Loading player data...")
    df = load_data(predictions_file, metadata_file)

    print("🧩 Optimizing team lineup...")
    selected_team = optimize_team(df, budget=budget, max_per_team=max_per_team)

    print(f"✔ Selected {len(selected_team)} players")

    # Convert to JSON
    selected_team.to_json(output_file, orient="records", indent=4)
    print(f"Optimized team saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=str, required=True, help="Predicted player scores (.npy)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Player metadata CSV (positions, salaries, team)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output JSON file for selected lineup"
    )
    parser.add_argument("--budget", type=int, default=100, help="Total squad budget")
    parser.add_argument(
        "--max_per_team", type=int, default=3, help="Max players allowed per real team"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_optimizer(
        args.features, args.metadata, args.output, args.budget, args.max_per_team
    )
