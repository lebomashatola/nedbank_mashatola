import argparse
import json
import os
import pandas as pd
import requests

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
def load_local(output_path):
    print("📁 Loading local CSV...")

    csv_path = "data/local_players.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Local CSV not found: data/local_players.csv\n"
            "Please place your dataset under /data."
        )

    df = pd.read_csv(csv_path)

    players = df.to_dict(orient="records")

    print(f"✔ Loaded {len(players)} players from local CSV")

    with open(output_path, "w") as f:
        json.dump(players, f, indent=4)

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

    print(f"🎉 Data acquisition complete → {args.output}")
