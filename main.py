import sys
import os
import argparse
import subprocess
from src.config import Config

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.append(os.path.join(ROOT, "src"))

# Load rest of modules
from src.data_loader import PlayerFetcher
from src.embeddings import (
    generate_embeddings,
)  


def fetch_player_data(cfg: Config):
    """
    Fetch or generate player data according to the config file.
    """
    print("Fetching / generating player data...")

    fetcher = PlayerFetcher(cfg=cfg)  

    if cfg.source.lower() == "tsdb":
        print("Using TheSportsDB API...")
        df = fetcher.fetch_tsdb()
    else:
        print("Generating synthetic dataset...")
        df = fetcher.generate_synthetic()

    print(f"Player dataframe loaded. Shape = {df.shape}")
    return df


def launch_streamlit_app():
    """
    Launch the Streamlit dashboard.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(project_root, "app.py")
    try:
        print(f"Launching Streamlit app: {app_path}")
        subprocess.run(["streamlit", "run", app_path], cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Streamlit failed to launch: {e}")


def main():
    """
    Main orchestration program:
    1. Load config
    2. Fetch or generate player data
    3. Generate embeddings (with logging)
    4. Launch Streamlit dashboard
    """

    # Ensure --config argument is parsed during main.py call
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Name of YAML config file inside configs/ directory"
    )
    args = parser.parse_args()

    print("===== Starting main.py =====")

    # Load global config
    cfg = Config(args.config)
    print("Config loaded successfully.")
    print(f"Full Configuration:\n{cfg.config}")

    # Step 1 — Fetch or generate data
    df_players = fetch_player_data(cfg)
    print("Player data ready.")

    # Step 2 — Generate embeddings
    if cfg.config.get("training", {}).get("generate_embeddings", False):
        generate_embeddings(cfg=cfg)
    else:
        print("Skipping embedding generation (disabled in config).")

    # Step 3 — Optimization + ML handled inside Streamlit app
    launch_streamlit_app()


if __name__ == "__main__":
    main()
