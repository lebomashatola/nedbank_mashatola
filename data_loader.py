import os
import yaml
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Any


class PlayerFetcher:
    """
    Class to fetch player data either from TheSportsDB API or generate synthetic data.
    Configuration is loaded from a YAML file that specifies leagues and data source.
    """

    # Mapping of league names to TheSportsDB league IDs
    TSDB_LEAGUES = {
        "EPL": "4328",
        "LaLiga": "4335",
        "SerieA": "4332",
        "Bundesliga": "4331",
    }

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the fetcher with configuration.
        
        Args:
            config_path: Path to the YAML config file.
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.base_dir, config_path)
        self.config = self.load_config()
        self.selected_leagues = self.config["data"].get("leagues", [])
        self.source = self.config["data"].get("source", "tsdb").lower()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def safe_get(self, url: str) -> Dict[str, Any]:
        """
        Perform a safe GET request and return JSON data.
        Returns empty dict if request fails.
        """
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[ERROR] Request failed: {url}\n{e}")
            return {}

    def fetch_tsdb(self) -> pd.DataFrame:
        """
        Fetch players from TheSportsDB for configured leagues.
        Returns a pandas DataFrame and saves CSV to 'data' folder.
        """
        for league in self.selected_leagues:
            if league not in self.TSDB_LEAGUES:
                raise ValueError(
                    f"Invalid league '{league}'. Valid options: {list(self.TSDB_LEAGUES.keys())}"
                )

        all_players = []

        # Loop through selected leagues
        for league_name in self.selected_leagues:
            league_id = self.TSDB_LEAGUES[league_name]
            print(f"\n[TSDB] Processing league: {league_name}")

            # Fetch all teams for the league
            teams_url = f"https://www.thesportsdb.com/api/v1/json/3/lookup_all_teams.php?id={league_id}"
            teams = self.safe_get(teams_url).get("teams", [])
            print(f"[TSDB] → {len(teams)} teams found")

            # Fetch players for each team
            for team in teams:
                team_id = team.get("idTeam")
                team_name = team.get("strTeam")
                if not team_id:
                    continue

                print(f"[TSDB] Fetching players for: {team_name}")
                players_url = f"https://www.thesportsdb.com/api/v1/json/3/lookup_all_players.php?id={team_id}"
                players = self.safe_get(players_url).get("player", [])
             
                # Add league and team info
                for p in players:
                    p["team"] = team_name
                    p["league"] = league_name
                    all_players.append(p)

        # Convert to DataFrame
        df = pd.DataFrame(all_players)
        
        # Save CSV in 'data' folder inside script directory
        self.save_csv(df, "tsdb_players.csv")
        print(f"[TSDB] Total players collected: {len(df)}")
        return df

    def generate_synthetic(self) -> pd.DataFrame:
        """
        Generate synthetic player statistics based on configuration parameters.
        
        Configuration parameters are loaded from the YAML config file 'synthtic' section:
        - n_players: Number of players to generate
        - n_weeks: Number of weeks of historical points
        - positions: List of player positions
        - seed: Random seed for reproducibility
        
        Returns:
            pandas DataFrame with synthetic player stats
        """
        # Load synthetic parameters from config
        synthetic_config = self.config.get("synthetic", {})
        n_players = synthetic_config.get("n_players", 200)
        n_weeks = synthetic_config.get("n_weeks", 10)
        positions = synthetic_config.get("positions", ["GK", "DEF", "MID", "FWD"])
        seed = synthetic_config.get("seed", 42)

        np.random.seed(seed)

        # Base points and salary by position
        base_points = {"GK": 3.5, "DEF": 4.0, "MID": 5.0, "FWD": 5.5}
        base_salary = {"GK": 5.0, "DEF": 6.0, "MID": 7.5, "FWD": 8.0}

        players = []
        for i in range(n_players):
            # Random position and player quality
            position = np.random.choice(positions)
            quality = np.clip(np.random.normal(1.0, 0.3), 0.5, 2.0)

            # Basic player info
            player_data = {
                "player_id": i,
                "name": f"Player_{i}",
                "position": position,
                "team": f"Team_{np.random.randint(0, 20)}",
                "salary": base_salary[position] * quality + np.random.normal(0, 0.5),
                "quality": quality,
            }

            # Generate points for each week
            base_pts = base_points[position] * quality
            for week in range(n_weeks):
                form = np.random.normal(0, 1.5)
                points = np.random.poisson(max(0, base_pts + form))
                player_data[f"points_week_{week}"] = points
            players.append(player_data)

        df = pd.DataFrame(players)
        df["salary"] = df["salary"].clip(4.0, 13.0)

        # Save CSV
        self.save_csv(df, "synthetic_players.csv")
        print(f"[Synthetic] Total players generated: {len(df)}")
        return df

    def save_csv(self, df: pd.DataFrame, filename: str):
        """
        Save DataFrame as CSV in the 'data' folder inside the script directory.
        Creates folder if it does not exist.
        """
        data_dir = os.path.join(self.base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, filename)
        df.to_csv(path, index=False)
        print(f"[Saved CSV] {path}")

    def fetch(self) -> pd.DataFrame:
        """
        Main method to fetch data based on source from config.
        Returns DataFrame from either TSDB or synthetic data.
        """
        if self.source == "tsdb":
            return self.fetch_tsdb()
        elif self.source == "synthetic":
            return self.generate_synthetic()
        else:
            raise ValueError(f"Invalid source '{self.source}'. Must be 'tsdb' or 'synthetic'.")

if __name__ == "__main__":
    fetcher = PlayerFetcher()
    df_players = fetcher.fetch()
    print(df_players.head())
