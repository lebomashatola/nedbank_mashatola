import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from src.config import Config
from src.logger import setup_logger

# Setup logger
logger = setup_logger("Embedding")


# Dataset wrapper for tabular data
class TabularDataset(Dataset):
    """PyTorch Dataset wrapper to handle tabular data with categorical and continuous features."""

    def __init__(
        self, df: pd.DataFrame, categorical_cols: List[str], continuous_cols: List[str]
    ):
        self.X_cat = (
            df[categorical_cols].values.astype(np.int64) if categorical_cols else None
        )
        self.X_cont = (
            df[continuous_cols].values.astype(np.float32) if continuous_cols else None
        )

    def __len__(self):
        return len(self.X_cont) if self.X_cont is not None else len(self.X_cat)

    def __getitem__(self, idx):
        cat = (
            torch.tensor(self.X_cat[idx])
            if self.X_cat is not None
            else torch.tensor([], dtype=torch.long)
        )
        cont = (
            torch.tensor(self.X_cont[idx])
            if self.X_cont is not None
            else torch.tensor([], dtype=torch.float32)
        )
        return cat, cont


# TabTransformer model for tabular embeddings
class TabTransformer(nn.Module):
    """Simple TabTransformer with embeddings, attention, and feed-forward network."""

    def __init__(
        self,
        cat_dims: List[int],
        cont_dim: int,
        embed_dim: int = 64,
        num_heads: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.cat_embeddings = (
            nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in cat_dims])
            if cat_dims
            else None
        )
        self.cont_bn = nn.BatchNorm1d(cont_dim) if cont_dim > 0 else None
        self.attention = (
            nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads, batch_first=True
            )
            if cat_dims
            else None
        )
        self.fc_cat = nn.Linear(embed_dim, embed_dim) if cat_dims else None
        self.fc_cont = nn.Linear(cont_dim, embed_dim) if cont_dim > 0 else None

        input_dim = (embed_dim if cat_dims else 0) + (embed_dim if cont_dim > 0 else 0)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor):
        embeddings = []

        if self.cat_embeddings and x_cat.numel() > 0:
            cat_emb_list = [
                emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)
            ]
            cat_emb = torch.stack(cat_emb_list, dim=1)
            att_out, _ = self.attention(cat_emb, cat_emb, cat_emb)
            att_out = att_out.mean(dim=1)
            embeddings.append(self.fc_cat(att_out))

        if self.cont_bn and x_cont.numel() > 0:
            embeddings.append(self.fc_cont(self.cont_bn(x_cont)))

        x = torch.cat(embeddings, dim=1)
        reconstructed = self.ffn(x)
        return x, reconstructed


# Function to generate embeddings
def generate_embeddings(cfg: Optional[Config] = None):
    if cfg is None:
        cfg = Config()
        logger.info("Loaded default config for embeddings.")

    # Paths
    data_dir = os.path.join(cfg.base_dir, "data")
    csv_file = os.path.join(data_dir, "synthetic_players.csv")
    emb_file = os.path.join(data_dir, "synthetic_embeddings_tabtransformer.csv")

    if not os.path.exists(csv_file):
        logger.error(f"Player CSV not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    logger.info(f"Loaded player data: {df.shape}")

    # Identify columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    continuous_cols = df.select_dtypes(
        include=["int64", "float64", "float32"]
    ).columns.tolist()

    cat_dims = []
    for col in categorical_cols:
        df[col], uniques = pd.factorize(df[col])
        cat_dims.append(len(uniques))

    dataset = TabularDataset(df, categorical_cols, continuous_cols)
    dataloader = DataLoader(
        dataset, batch_size=cfg.training.get("batch_size", 64), shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = TabTransformer(
        cat_dims=cat_dims,
        cont_dim=len(continuous_cols),
        embed_dim=cfg.model_config.get("embedding_dim", 64),
        num_heads=cfg.model_config.get("num_heads", 8),
        hidden_dim=cfg.model_config.get("hidden_dim", 128),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.get("learning_rate", 1e-3)
    )
    criterion = nn.MSELoss()

    epochs = cfg.training.get("epochs", 50)
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for x_cat, x_cont in dataloader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            optimizer.zero_grad()
            embeddings, reconstructed = model(x_cat, x_cont)

            target = torch.zeros_like(embeddings)
            if x_cont.numel() > 0:
                target[:, -x_cont.size(1) :] = x_cont

            loss = criterion(reconstructed, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (x_cont.size(0) if x_cont.numel() > 0 else 1)

        total_loss /= len(dataset)
        if epoch == 1 or epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs}, Loss: {total_loss:.6f}")

    # Save embeddings
    model.eval()
    all_cat = (
        torch.tensor(df[categorical_cols].values.astype(np.int64), device=device)
        if categorical_cols
        else torch.tensor([])
    )
    all_cont = (
        torch.tensor(df[continuous_cols].values.astype(np.float32), device=device)
        if continuous_cols
        else torch.tensor([])
    )

    with torch.no_grad():
        embeddings, _ = model(all_cat, all_cont)
        emb_df = pd.DataFrame(embeddings.cpu().numpy(), index=df.index)
        emb_df.to_csv(emb_file, index=False)
        logger.info(f"Saved embeddings → {emb_file}")


if __name__ == "__main__":
    cfg = Config()
    generate_embeddings(cfg)
