import os
import yaml


class Config:
    """
    Configuration loader and wrapper for the football player performance system.
    Automatically loads YAML config and provides easy access to all sections.
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Base directory = project root (parent of src)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.base_dir, "configs", config_path)

        # Load YAML config
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Convenience attributes for top-level sections
        self.data = self.config.get("data", {})
        self.synthetic = self.config.get("synthetic", {})
        self.training = self.config.get("training", {})
        self.model_config = self.config.get("model", {})
        self.optimizer = self.config.get("optimizer", {})
        self.ml_model = self.config.get("ml_model", {})
        self.opponent_features = self.config.get("opponent_features", {})

        # Quick-access attributes
        self.selected_leagues = self.data.get("leagues", [])
        self.source = self.data.get("source", "tsdb").lower()
