# main.py
import os
import subprocess
from data_loader import PlayerFetcher
from lineup_generator import Config, generate_lineups
from embeddings import main as tabtransformer_main


def fetch_player_data(cfg):
    """
    Fetch or generate player data based on configuration.
    """
    fetcher = PlayerFetcher(cfg.config_path)
    
    if cfg.opponent_features.get("use_tsdb", False):
        print("[INFO] Fetching data from TheSportsDB...")
        df = fetcher.fetch_tsdb()
    else:
        print("[INFO] Generating synthetic player data...")
        df = fetcher.generate_synthetic()
    
    print(f"[INFO] Data shape: {df.shape}")
    print(df.head())
    return df

def generate_embeddings():
    """
    Run the TabTransformer embeddings extraction.
    """
    print("[INFO] Generating embeddings via TabTransformer...")
    tabtransformer_main()

def launch_streamlit_app():
    """
    Launch the Streamlit dashboard.
    """
    streamlit_file = os.path.join(os.path.dirname(__file__), "app.py")
    print(f"[INFO] Launching Streamlit app → {streamlit_file}")
    subprocess.run(["streamlit", "run", streamlit_file])

def main():
    # Load config
    cfg = Config("config.yaml")
    
    # Step 1: Fetch or generate player data
    df_players = fetch_player_data(cfg)
    
    # Step 2: Generate embeddings (optional)
    if cfg.config.get("training", {}).get("generate_embeddings", True):
        generate_embeddings()

    # Step 3: Lineups triggered inside Streamlist and dashboard lanched 
    launch_streamlit_app()

if __name__ == "__main__":
    main()
