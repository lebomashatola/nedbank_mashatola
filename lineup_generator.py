# lineup_generator.py

import os
import yaml
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.base_dir, config_path)
        self.config = self.load_config()

        optimizer_cfg = self.config.get("optimizer", {})
        self.budget = optimizer_cfg.get("budget", 100)
        self.roster_constraints = optimizer_cfg.get("positions", {"GK":1,"DEF":4,"MID":4,"FWD":2})

        ml_cfg = self.config.get("ml_model", {})
        self.model_type = ml_cfg.get("type","RandomForest")
        self.hyperparams = ml_cfg.get("hyperparams",{})
        self.random_state = ml_cfg.get("random_state",42)
        self.tune = ml_cfg.get("tune",True)

        self.opponent_features = self.config.get("opponent_features", {})

    def load_config(self):
        with open(self.config_path,"r") as f:
            return yaml.safe_load(f)

def generate_lineups(opponent_features=None):
    cfg = Config()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    df = pd.read_csv(os.path.join(data_dir, "synthetic_players.csv"))
    emb_df = pd.read_csv(os.path.join(data_dir, "synthetic_embeddings_tabtransformer.csv"))

    if "salary" not in df.columns:
        df["salary"] = np.random.uniform(4,13,size=len(df))
    if "points_week_0" in df.columns:
        y = df["points_week_0"].values
    else:
        y = np.random.uniform(0,10,size=len(df))

    # Incorporate opponent features
    if opponent_features is not None:
        opp_df = pd.DataFrame([opponent_features]*len(df))
        X_raw = np.hstack([emb_df.values, opp_df.values])
    elif cfg.opponent_features:
        opp_df = pd.DataFrame([cfg.opponent_features]*len(df))
        X_raw = np.hstack([emb_df.values, opp_df.values])
    else:
        X_raw = emb_df.values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # ML model
    if cfg.model_type.lower()=="randomforest":
        base_model = RandomForestRegressor(random_state=cfg.random_state)
    elif cfg.model_type.lower()=="gradientboosting":
        base_model = GradientBoostingRegressor(random_state=cfg.random_state)
    else:
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")

    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=cfg.random_state)

    if cfg.tune and cfg.hyperparams:
        grid_search = GridSearchCV(base_model, cfg.hyperparams, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        base_model.fit(X_train, y_train)
        best_model = base_model

    best_model.fit(X, y)
    df["predicted_points"] = best_model.predict(X)

    # LP optimization
    prob = LpProblem("FantasyTeamSelection", LpMaximize)
    player_vars = LpVariable.dicts("select", df.index, cat=LpBinary)
    prob += lpSum(df.loc[i,"predicted_points"]*player_vars[i] for i in df.index)
    prob += lpSum(df.loc[i,"salary"]*player_vars[i] for i in df.index) <= cfg.budget
    for pos,count in cfg.roster_constraints.items():
        prob += lpSum(player_vars[i] for i in df.index if df.loc[i,"position"]==pos)==count
    prob.solve()
    ml_lineup = df[[player_vars[i].varValue==1 for i in df.index]].copy()
    ml_lineup = ml_lineup.sort_values("predicted_points",ascending=False)
    ml_points = ml_lineup["predicted_points"].sum()

    # Baselines
    def random_baseline(df, cfg):
        while True:
            selected = df.sample(sum(cfg.roster_constraints.values()))
            valid = True
            for pos,count in cfg.roster_constraints.items():
                if (selected['position']==pos).sum()!=count:
                    valid=False
                    break
            if selected['salary'].sum()>cfg.budget:
                valid=False
            if valid:
                return selected

    def top_historical_baseline(df, cfg):
        sort_col = "points_week_0" if "points_week_0" in df.columns else "predicted_points"
        df_sorted = df.sort_values(sort_col, ascending=False)
        selected = pd.DataFrame()
        for pos,count in cfg.roster_constraints.items():
            selected = pd.concat([selected, df_sorted[df_sorted['position']==pos].head(count)])
        if selected['salary'].sum()>cfg.budget:
            selected = selected.sort_values(sort_col,ascending=False)
            total_salary=0
            final_selected=pd.DataFrame()
            for idx,row in selected.iterrows():
                if total_salary+row['salary']<=cfg.budget:
                    final_selected=pd.concat([final_selected,row.to_frame().T])
                    total_salary+=row['salary']
            selected = final_selected
        return selected

    # Compute baselines
    random_points_list = [random_baseline(df,cfg)["predicted_points"].sum() for _ in range(100)]
    mean_random_points = np.mean(random_points_list)
    top_lineup = top_historical_baseline(df,cfg)
    top_points = top_lineup["predicted_points"].sum()

    return ml_lineup, ml_points, top_lineup, mean_random_points, top_points
