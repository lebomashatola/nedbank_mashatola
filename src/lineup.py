import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary
from src.config import Config


def generate_lineups(opponent_features=None, cfg: Config = None):
    """
    Generate optimal fantasy football lineup using ML predictions and LP optimization.
    Also computes random and top historical baseline lineups.

    Args:
        opponent_features (dict, optional): Features of the upcoming opponent team.
        cfg (Config, optional): Configuration object.

    Returns:
        Tuple: (ML optimal lineup, ML points, top historical lineup,
                mean random lineup points, top historical points)
    """
    if cfg is None:
        cfg = Config()
        print("Loaded default config for lineup generation.")

    # Base directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    # Load player and embeddings data
    df_file = os.path.join(data_dir, "synthetic_players.csv")
    emb_file = os.path.join(data_dir, "synthetic_embeddings_tabtransformer.csv")

    if not os.path.exists(df_file) or not os.path.exists(emb_file):
        print("Player CSV or embeddings CSV not found.")
        return None, None, None, None, None

    df = pd.read_csv(df_file)
    emb_df = pd.read_csv(emb_file)
    print(f"Loaded {df.shape[0]} players and embeddings of shape {emb_df.shape}.")

    # Ensure salary column exists
    if "salary" not in df.columns:
        df["salary"] = np.random.uniform(4, 13, size=len(df))

    # Target variable: points for week 0 or synthetic points
    y = (
        df["points_week_0"].values
        if "points_week_0" in df.columns
        else np.random.uniform(0, 10, len(df))
    )

    # Combine embeddings with opponent features if provided
    if opponent_features:
        opp_df = pd.DataFrame([opponent_features] * len(df))
        X_raw = np.hstack([emb_df.values, opp_df.values])
    elif cfg.opponent_features:
        opp_df = pd.DataFrame([cfg.opponent_features] * len(df))
        X_raw = np.hstack([emb_df.values, opp_df.values])
    else:
        X_raw = emb_df.values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # ML model selection
    model_cfg = cfg.ml_model
    opt_cfg = cfg.optimizer

    model_type = model_cfg["type"].lower()
    if model_type == "randomforest":
        base_model = RandomForestRegressor(random_state=model_cfg["random_state"])
    elif model_type == "gradientboosting":
        base_model = GradientBoostingRegressor(random_state=model_cfg["random_state"])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train-test split for hyperparameter tuning if required
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=model_cfg["random_state"]
    )

    if model_cfg.get("tune") and model_cfg.get("hyperparams"):
        # Perform GridSearch for best hyperparameters
        grid_search = GridSearchCV(
            base_model,
            model_cfg["hyperparams"],
            scoring="neg_mean_squared_error",
            cv=3,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best hyperparameters found: {grid_search.best_params_}")
    else:
        base_model.fit(X_train, y_train)
        best_model = base_model

    # Fit final model on full dataset
    best_model.fit(X, y)
    df["predicted_points"] = best_model.predict(X)

    # -------------------------
    # LP Optimization to select best lineup
    # -------------------------
    prob = LpProblem("FantasyTeamSelection", LpMaximize)
    player_vars = LpVariable.dicts("select", df.index, cat=LpBinary)

    # Objective: maximize predicted points
    prob += lpSum(df.loc[i, "predicted_points"] * player_vars[i] for i in df.index)

    # Constraint: total salary within budget
    prob += (
        lpSum(df.loc[i, "salary"] * player_vars[i] for i in df.index)
        <= opt_cfg["budget"]
    )

    # Constraints: number of players per position
    for pos, count in opt_cfg["positions"].items():
        prob += (
            lpSum(player_vars[i] for i in df.index if df.loc[i, "position"] == pos)
            == count
        )

    # Solve LP
    prob.solve()

    # Extract ML-optimal lineup
    ml_lineup = df[[player_vars[i].varValue == 1 for i in df.index]].copy()
    ml_lineup = ml_lineup.sort_values("predicted_points", ascending=False)
    ml_points = ml_lineup["predicted_points"].sum()
    print(f"ML optimal lineup points: {ml_points:.2f}")

    # -------------------------
    # Baseline lineups
    # -------------------------

    def random_baseline(df, cfg):
        """Generate a random valid lineup within budget and position constraints"""
        while True:
            selected = df.sample(sum(opt_cfg["positions"].values()))
            valid = True
            for pos, count in opt_cfg["positions"].items():
                if (selected["position"] == pos).sum() != count:
                    valid = False
                    break
            if selected["salary"].sum() > cfg.optimizer["budget"]:
                valid = False
            if valid:
                return selected

    def top_historical_baseline(df, cfg):
        """Select the top historical points lineup respecting budget and positions"""
        sort_col = (
            "points_week_0" if "points_week_0" in df.columns else "predicted_points"
        )
        df_sorted = df.sort_values(sort_col, ascending=False)
        selected = pd.DataFrame()
        for pos, count in opt_cfg["positions"].items():
            selected = pd.concat(
                [selected, df_sorted[df_sorted["position"] == pos].head(count)]
            )
        # Adjust for budget constraint if needed
        if selected["salary"].sum() > cfg.optimizer["budget"]:
            selected = selected.sort_values(sort_col, ascending=False)
            total_salary = 0
            final_selected = pd.DataFrame()
            for idx, row in selected.iterrows():
                if total_salary + row["salary"] <= cfg.optimizer["budget"]:
                    final_selected = pd.concat([final_selected, row.to_frame().T])
                    total_salary += row["salary"]
            selected = final_selected
        return selected

    # Compute baseline points
    random_points_list = [
        random_baseline(df, cfg)["predicted_points"].sum() for _ in range(100)
    ]
    mean_random_points = np.mean(random_points_list)
    top_lineup = top_historical_baseline(df, cfg)
    top_points = top_lineup["predicted_points"].sum()

    print(f"Random baseline mean points: {mean_random_points:.2f}")
    print(f"Top historical lineup points: {top_points:.2f}")

    return ml_lineup, ml_points, top_lineup, mean_random_points, top_points


if __name__ == "__main__":
    generate_lineups()
