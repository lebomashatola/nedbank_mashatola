import argparse
import json
import numpy as np
import os
import pandas as pd
import requests
from typing import List
import yaml

# -------------------------
# Loaders (unchanged)
# -------------------------


def load_espn(output_path):
    print("Fetching ESPN public data...")
    url = "https://site.api.espn.com/apis/v2/sports/soccer/eng.1/athletes"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"ESPN API returned error {resp.status_code}")
    data = resp.json()
    player_entries = data.get("items", [])
    players = []
    for p in player_entries:
        players.append(
            {
                "id": p.get("id"),
                "name": p.get("displayName"),
                "team": p.get("team", {}).get("displayName"),
                "position": p.get("position", {}).get("displayName"),
                "nationality": p.get("nationality", "Unknown"),
                "stats": p.get("statistics", []),
                "age": p.get("age"),
            }
        )
    print(f"Retrieved {len(players)} players from ESPN")
    with open(output_path, "w") as f:
        json.dump(players, f, indent=4)

    return players


def load_thesportsdb(output_path):
    print("Fetching data from TheSportsDB...")
    url = "https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?t=Arsenal"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"TheSportsDB API returned error {resp.status_code}")
    json_data = resp.json()
    players_raw = json_data.get("player", [])
    players = []
    for p in players_raw:
        players.append(
            {
                "id": p.get("idPlayer"),
                "name": p.get("strPlayer"),
                "team": p.get("strTeam"),
                "position": p.get("strPosition"),
                "nationality": p.get("strNationality"),
                "height": p.get("strHeight"),
                "weight": p.get("strWeight"),
                "description": p.get("strDescriptionEN"),
            }
        )
    print(f"Retrieved {len(players)} players from TheSportsDB")
    with open(output_path, "w") as f:
        json.dump(players, f, indent=4)

    return players


def load_local(
    output_path,
    n_players: int = 200,
    n_weeks: int = 10,
    positions: List[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    if positions is None:
        positions = ["GK", "DEF", "MID", "FWD"]

    position_base_points = {"GK": 3.5, "DEF": 4.0, "MID": 5.0, "FWD": 5.5}
    position_base_salary = {"GK": 5.0, "DEF": 6.0, "MID": 7.5, "FWD": 8.0}
    players = []

    for i in range(n_players):
        position = np.random.choice(positions)
        quality = np.clip(np.random.normal(1.0, 0.3), 0.5, 2.0)
        player_data = {
            "player_id": i,
            "name": f"Player_{i}",
            "position": position,
            "team": f"Team_{np.random.randint(0, 20)}",
            "salary": position_base_salary[position] * quality
            + np.random.normal(0, 0.5),
            "quality": quality,
        }

        base_points = position_base_points[position] * quality
        for week in range(n_weeks):
            form = np.random.normal(0, 1.5)
            points = np.random.poisson(max(0, base_points + form))
            player_data[f"points_week_{week}"] = points
        players.append(player_data)

    players = pd.DataFrame(players)
    players["salary"] = players["salary"].clip(4.0, 13.0)
    players.to_json(output_path, orient="records", indent=4)

    return players


# -------------------------
# CONFIG FILE LOADER
# -------------------------
def load_config(config_path: str):
    with open(config_path, "r") as f:
        if config_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        else:
            return json.load(f)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file (JSON or YAML)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = config.get("data", {})

    source = dataset_config.get("source", "local")
    output = dataset_config.get("output", "data/players.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"Using dataset source: {source}")
    print(f"Output path: {output}")

    if source == "espn":
        load_espn(output)
    elif source == "thesportsdb":
        load_thesportsdb(output)
    else:
        local_params = dataset_config.get("local_params", {})
        load_local(output_path=output, **local_params)

    print(f"Data acquisition complete → {output}")
