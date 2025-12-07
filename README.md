# Fantasy Football Player Performance System

This repository provides a **Fantasy Football Player Performance System** leveraging modern machine learning techniques. It enables you to **fetch player data**, **generate embeddings** using **Tabular Transformer models**, **predict player performance** with ensemble ML models, **optimize lineups**, and **visualize results** via a **Streamlit dashboard**.

---

## Features

- **Data Acquisition**: Fetch real player data from TheSportsDB API or generate synthetic datasets for testing.
- **Embeddings Generation**: Generate rich feature embeddings from categorical and continuous player data using **Tabular Transformer (TabTransformer)** models.
- **Machine Learning Models**: Predict player performance using:
  - RandomForest
  - GradientBoosting (optional)
- **Lineup Optimization**: Select optimal fantasy football lineups based on predicted points, salary constraints, and positional requirements using **linear programming**.
- **Visualization**: Explore player stats and generated lineups interactively with **Streamlit**.
- **Unified Logging**: Each run generates a dedicated log file recording data loading, model training, and embedding generation progress.

---


---

## Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/lebomashatola/fantasy-football.git
cd fantasy-football
```

### 2. Create conda environment

```bash
conda create -n fantasy-env python=3.11 -y
conda activate fantasy-env
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Copy or create configs/config.yaml in the root directory. Example sections:

```yaml
data:
data:
  leagues: ["EPL", "LaLiga"]
  source: "tsdb"

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 50

model:
  embedding_dim: 64
  num_heads: 8
  hidden_dim: 128

ml_model:
  type: "random_forest"
  params:
    n_estimators: 100
    max_depth: 10
```

### 5. Run the script
```bash
python main.py 
```
OR 
```bash
bash run_main.sh
```

### 6. Featch player data 
```python
from src.config import Config
from src.player_data import fetch_player_data

cfg = Config()
fetch_player_data(cfg)
```

### 7. Launch Streamlit App
```python
from src.streamlit_app import launch_streamlit_app

launch_streamlit_app()
```

### 8. Generate Embeddings
```python
from src.embeddings import generate_embeddings
from src.config import Config

cfg = Config()
generate_embeddings(cfg)
```

### 9. optimize Lineup
```python
from src.lineup_generator import generate_lineups
from src.config import Config

cfg = Config()
lineups = generate_lineups(cfg)
print(lineups)
```

### 10. Streamlit
```python
from src.streamlit_app import launch_streamlit_app

launch_streamlit_app()
```

### Logging Info
```yaml
2025-12-07 06:00:12 | Embedding | INFO | Loaded player data: (1000, 20)
2025-12-07 06:00:12 | Embedding | INFO | Using device: cuda
2025-12-07 06:01:00 | Embedding | INFO | Epoch 10/50, Loss: 0.012345
2025-12-07 06:05:00 | Embedding | INFO | Saved embeddings → data/synthetic_embeddings_tabtransformer.csv
```

