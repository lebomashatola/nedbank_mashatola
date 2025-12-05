import argparse
import json
import numpy as np
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Tuple

"""
DATA SOURCES SUPPORTED
----------------------
1. ESPN PUBLIC DATA (no API key required)
   - Uses public JSON feed from ESPN Fantasy / Stats
   - Example endpoint: https://site.api.espn.com/apis/v2/sports/soccer/eng.1/athletes

2. TheSportsDB (Free Tier)
   - Example endpoint: https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p=Messi

3. LOCAL CSV
   - User-provided historical dataset
"""


# -------------------------
# ESPN API LOADER
# -------------------------
def load_espn(output_path):
    print("📡 Fetching ESPN public data...")

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

    print(f"✔ Retrieved {len(players)} players from ESPN")

    with open(output_path, "w") as f:
        json.dump(players, f, indent=4)

    return players


# -------------------------
# TheSportsDB LOADER
# -------------------------
def load_thesportsdb(output_path):
    print("📡 Fetching data from TheSportsDB...")

    # English Premier League player list endpoint (free)
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

    print(f"✔ Retrieved {len(players)} players from TheSportsDB")

    with open(output_path, "w") as f:
        json.dump(players, f, indent=4)

    return players


# -------------------------
# LOCAL CSV LOADER
# -------------------------
def load_local(
    n_players: int = 200, n_weeks: int = 10, positions: List[str] = None, seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic player statistics for fantasy sports

    Args:
        n_players: Number of players
        n_weeks: Number of weeks of historical data
        positions: Player positions (e.g., ['GK', 'DEF', 'MID', 'FWD'])

    Returns:
        DataFrame with player stats across multiple weeks
    """
    np.random.seed(seed)

    if positions is None:
        positions = ["GK", "DEF", "MID", "FWD"]

    # Base stats by position
    position_base_points = {"GK": 3.5, "DEF": 4.0, "MID": 5.0, "FWD": 5.5}

    position_base_salary = {"GK": 5.0, "DEF": 6.0, "MID": 7.5, "FWD": 8.0}

    players = []
    for i in range(n_players):
        position = np.random.choice(positions)

        # Player quality (some players are just better)
        quality = np.random.normal(1.0, 0.3)
        quality = np.clip(quality, 0.5, 2.0)

        player_data = {
            "player_id": i,
            "name": f"Player_{i}",
            "position": position,
            "team": f"Team_{np.random.randint(0, 20)}",
            "salary": position_base_salary[position] * quality
            + np.random.normal(0, 0.5),
            "quality": quality,
        }

        # Generate weekly points
        base_points = position_base_points[position] * quality
        for week in range(n_weeks):
            # Add form (recent performance affects future)
            form = np.random.normal(0, 1.5)
            points = np.random.poisson(max(0, base_points + form))
            player_data[f"points_week_{week}"] = points

        players.append(player_data)

    players = pd.DataFrame(players)
    players["salary"] = players["salary"].clip(4.0, 13.0)

    return players


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source", type=str, required=True, choices=["espn", "thesportsdb", "local"]
    )
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.source == "espn":
        load_espn(args.output)
    elif args.source == "thesportsdb":
        load_thesportsdb(args.output)
    else:
        load_local(args.output)

    print(f"Data acquisition complete → {args.output}")
