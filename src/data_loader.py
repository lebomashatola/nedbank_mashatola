import os
import yaml
import requests
import numpy as np
import pandas as pd


class PlayerFetcher:
    """
    Fetch or generate player data.
    """

    TSDB_LEAGUES = {
        "EPL": "4328",
        "LaLiga": "4335",
        "SerieA": "4332",
        "Bundesliga": "4331",
    }

    def __init__(self, cfg=None, config_path: str = "config.yaml"):
        print("Initializing PlayerFetcher...")

        # Config setup
        if cfg:
            self.config = cfg.config
            self.data = cfg.data
            self.synthetic = cfg.synthetic
            self.selected_leagues = cfg.selected_leagues
            self.source = cfg.source
            self.opponent_features = cfg.opponent_features
        else:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.config_path = os.path.join(self.base_dir, "configs", config_path)
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.data = self.config.get("data", {})
            self.synthetic = self.config.get("synthetic", {})
            self.selected_leagues = self.data.get("leagues", [])
            self.source = self.data.get("source", "synthetic")
            self.opponent_features = self.config.get("opponent_features", {})

        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Data directory set to {self.data_dir}")

    def safe_get(self, url: str):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            print(f"Fetched data from {url}")
            return resp.json()
        except Exception as e:
            print(f"Request failed: {url}\n{e}")
            return {}

    def fetch_tsdb(self) -> pd.DataFrame:
        print("Fetching player data from TheSportsDB API...")
        all_players = []

        for league_name in self.selected_leagues:
            league_id = self.TSDB_LEAGUES.get(league_name)
            if not league_id:
                print(f"Invalid league '{league_name}' skipped")
                continue

            teams_url = f"https://www.thesportsdb.com/api/v1/json/3/lookup_all_teams.php?id={league_id}"
            teams = self.safe_get(teams_url).get("teams", [])
            print(f"{len(teams)} teams found in {league_name}")

            for team in teams:
                team_id = team.get("idTeam")
                team_name = team.get("strTeam")
                if not team_id:
                    continue

                players_url = f"https://www.thesportsdb.com/api/v1/json/3/lookup_all_players.php?id={team_id}"
                players = self.safe_get(players_url).get("player", [])

                for p in players:
                    p["team"] = team_name
                    p["league"] = league_name
                    all_players.append(p)

        df = pd.DataFrame(all_players)
        self.save_csv(df, "tsdb_players.csv")
        print(f"Saved {len(df)} players to tsdb_players.csv")
        return df

    def generate_synthetic(self) -> pd.DataFrame:
        print("Generating synthetic player data...")
        n_players = self.synthetic.get("n_players", 200)
        n_weeks = self.synthetic.get("n_weeks", 10)
        positions = self.synthetic.get("positions", ["GK", "DEF", "MID", "FWD"])
        seed = self.synthetic.get("seed", 42)

        np.random.seed(seed)
        base_points = {"GK": 3.5, "DEF": 4.0, "MID": 5.0, "FWD": 5.5}
        base_salary = {"GK": 5.0, "DEF": 6.0, "MID": 7.5, "FWD": 8.0}

        players = []
        for i in range(n_players):
            position = np.random.choice(positions)
            quality = np.clip(np.random.normal(1.0, 0.3), 0.5, 2.0)
            player_data = {
                "player_id": i,
                "name": f"Player_{i}",
                "position": position,
                "team": f"Team_{np.random.randint(0, 20)}",
                "salary": base_salary[position] * quality + np.random.normal(0, 0.5),
                "quality": quality,
            }
            base_pts = base_points[position] * quality
            for week in range(n_weeks):
                form = np.random.normal(0, 1.5)
                points = np.random.poisson(max(0, base_pts + form))
                player_data[f"points_week_{week}"] = points
            players.append(player_data)

        df = pd.DataFrame(players)
        df["salary"] = df["salary"].clip(4.0, 13.0)
        self.save_csv(df, "synthetic_players.csv")
        print(f"Saved {len(df)} synthetic players to synthetic_players.csv")
        return df

    def save_csv(self, df: pd.DataFrame, filename: str):
        path = os.path.join(self.data_dir, filename)
        df.to_csv(path, index=False)
        print(f"DataFrame saved to {path}")

    def fetch(self) -> pd.DataFrame:
        print(f"Fetching data using source: {self.source}")
        if self.source.lower() == "tsdb":
            return self.fetch_tsdb()
        elif self.source.lower() == "synthetic":
            return self.generate_synthetic()
        else:
            raise ValueError(
                f"Invalid source '{self.source}'. Must be 'tsdb' or 'synthetic'."
            )
