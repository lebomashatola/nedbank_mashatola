import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from typing import List, Dict, Any

# Config file loader

class Config:
    """
    Loads hyperparameters from a YAML config file.
    Default config_path is 'config.yaml'.
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.base_dir, config_path)
        self.config = self.load_config()
        self.epochs = self.config["training"].get("epochs", 50)
        self.lr = self.config["training"].get("learning_rate", 1e-3)
        self.batch_size = self.config["training"].get("batch_size", 64)
        self.embed_dim = self.config["model"].get("embedding_dim", 64)
        self.num_heads = self.config["model"].get("num_heads", 8)
        self.hidden_dim = self.config["model"].get("hidden_dim", 128)

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

# Load dataset

class TabularDataset(Dataset):
    """
    PyTorch Dataset wrapper for tabular data.
    Separates categorical and continuous features.
    """
    def __init__(self, df, categorical_cols, continuous_cols):
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.X_cat = df[categorical_cols].values.astype(np.int64) if categorical_cols else None
        self.X_cont = df[continuous_cols].values.astype(np.float32) if continuous_cols else None

    def __len__(self):
        return len(self.X_cont) if self.X_cont is not None else len(self.X_cat)

    def __getitem__(self, idx):
        cat = torch.tensor(self.X_cat[idx]) if self.X_cat is not None else torch.tensor([], dtype=torch.long)
        cont = torch.tensor(self.X_cont[idx]) if self.X_cont is not None else torch.tensor([], dtype=torch.float32)
        return cat, cont

# TabTransformer Model

class TabTransformer(nn.Module):
    """
    TabTransformer-style autoencoder for tabular data.
    - Embeds categorical columns
    - Projects continuous columns
    - Applies multi-head self-attention on categorical embeddings
    - Concatenates embeddings and passes through feedforward network (FFN)
    """
    def __init__(self, cat_dims, cont_dim, embed_dim=64, num_heads=8, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Embedding layers for categorical columns
        self.cat_embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in cat_dims]) if cat_dims else None

        # BatchNorm for continuous features
        self.cont_bn = nn.BatchNorm1d(cont_dim) if cont_dim > 0 else None

        # Multi-head attention over categorical embeddings
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True) if cat_dims else None
        
        # Linear layers to project embeddings
        self.fc_cat = nn.Linear(embed_dim, embed_dim) if cat_dims else None
        self.fc_cont = nn.Linear(cont_dim, embed_dim) if cont_dim > 0 else None

        # Feedforward network for reconstruction (autoencoder)
        input_dim = 0
        if self.cat_embeddings:
            input_dim += embed_dim
        if cont_dim > 0:
            input_dim += embed_dim

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # reconstruct input
        )

    def forward(self, x_cat, x_cont):
        embeddings = []

        # Process categorical embeddings through attention
        if self.cat_embeddings and x_cat.numel() > 0:
            cat_emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            cat_emb = torch.stack(cat_emb_list, dim=1)  # shape: [batch, num_cat, embed_dim]
            att_out, _ = self.attention(cat_emb, cat_emb, cat_emb)
            att_out = att_out.mean(dim=1)  # pool across categorical features
            cat_emb_final = self.fc_cat(att_out)
            embeddings.append(cat_emb_final)

        # Process continuous features
        if self.cont_bn and x_cont.numel() > 0:
            x_cont_norm = self.cont_bn(x_cont)
            cont_emb = self.fc_cont(x_cont_norm)
            embeddings.append(cont_emb)
        
        # Concatenate embeddings
        x = torch.cat(embeddings, dim=1)

        # Reconstruct input via feedforward network
        reconstructed = self.ffn(x)

        # Return embeddings and reconstruction
        return x, reconstructed 

# Main
def main():

    # Load configuration
    cfg = Config()

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    csv_path = os.path.join(data_dir, "synthetic_players.csv")
    emb_path = os.path.join(data_dir, "synthetic_embeddings_tabtransformer.csv")

    # Load dataset
    df = pd.read_csv(csv_path)

    # Identify categorical and continuous columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    continuous_cols = df.select_dtypes(include=["int64", "float64", "float32"]).columns.tolist()

    # Factorize categorical columns to integer indices
    cat_dims = []
    for col in categorical_cols:
        df[col], uniques = pd.factorize(df[col])
        cat_dims.append(len(uniques))

    # Dataset and DataLoader
    dataset = TabularDataset(df, categorical_cols, continuous_cols)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabTransformer(cat_dims, len(continuous_cols), embed_dim=cfg.embed_dim,
                           num_heads=cfg.num_heads, hidden_dim=cfg.hidden_dim).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0
        for x_cat, x_cont in dataloader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            optimizer.zero_grad()
            embeddings, reconstructed = model(x_cat, x_cont)

            # Prepare target: continuous + zeros for categorical
            target = torch.cat([x_cat.float() * 0, x_cont], dim=1) if x_cat.numel() > 0 else x_cont
            if target.size(1) != reconstructed.size(1):
                target = torch.zeros_like(reconstructed)
                if x_cont.numel() > 0:
                    target[:, -x_cont.size(1):] = x_cont

            loss = criterion(reconstructed, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_cont.size(0)
        total_loss /= len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{cfg.epochs}, Loss: {total_loss:.6f}")

    # Extract embeddings
    model.eval()
    all_cat = torch.tensor(df[categorical_cols].values.astype(np.int64), device=device) if categorical_cols else torch.tensor([])
    all_cont = torch.tensor(df[continuous_cols].values.astype(np.float32), device=device) if continuous_cols else torch.tensor([])

    with torch.no_grad():
        embeddings, _ = model(all_cat, all_cont)
        emb_df = pd.DataFrame(embeddings.cpu().numpy(), index=df.index)
        emb_df.to_csv(emb_path, index=False)
        print(f"Saved embeddings → {emb_path}")

# -------------------------
if __name__ == "__main__":
    main()
