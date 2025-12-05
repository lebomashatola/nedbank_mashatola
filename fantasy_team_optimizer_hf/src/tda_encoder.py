import argparse
import os
import numpy as np
import pandas as pd
import dcor
from gudhi import RipsComplex, PersistenceLandscape

"""
TDA ENCODER WITH CSV INPUT & DISTANCE CORRELATION
------------------------------------------------

Input: CSV with numeric player features
Output: 
1. NumPy array of persistence landscape features (players x features)
2. Distance correlation matrix between features
"""


# -------------------------
# Convert row to point cloud
# -------------------------
def row_to_pointcloud(row, numeric_cols):
    """
    Convert player's numeric features into a point cloud
    """
    return row[numeric_cols].values.reshape(-1, 1)


# -------------------------
# Compute persistence landscape
# -------------------------
def compute_landscape(point_cloud, homology_dim=0, resolution=100):
    rips = RipsComplex(points=point_cloud)
    st = rips.create_simplex_tree(max_dimension=homology_dim + 1)
    diag = st.persistence(homology_coeff_field=2, min_persistence=0.01)
    # Filter by homology dimension
    diag_h = [pair[1] for pair in diag if pair[0] == homology_dim]
    if not diag_h:
        diag_h = np.array([[0.0, 0.0]])
    else:
        diag_h = np.array(diag_h)
    pl = PersistenceLandscape(resolution=resolution)
    pl.fit([diag_h])
    landscape_vec = pl.compute_landscape().flatten()
    return landscape_vec


# -------------------------
# Distance correlation matrix
# -------------------------
def compute_distance_correlation(df, numeric_cols):
    n = len(numeric_cols)
    corr_matrix = np.zeros((n, n))
    for i, col_i in enumerate(numeric_cols):
        for j, col_j in enumerate(numeric_cols):
            corr_matrix[i, j] = dcor.distance_correlation(
                df[col_i].values, df[col_j].values
            )
    return corr_matrix


# -------------------------
# Main TDA pipeline
# -------------------------
def run_tda(csv_path, tda_output, corr_output):
    print(f"📥 Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)

    # Identify numeric columns automatically
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric columns used for TDA: {numeric_cols}")

    # Compute distance correlation matrix
    print("🔹 Computing distance correlation matrix...")
    corr_matrix = compute_distance_correlation(df, numeric_cols)
    np.save(corr_output, corr_matrix)
    print(f"✔ Distance correlation matrix saved to {corr_output}")

    # Compute persistence landscapes for each player
    print("🔹 Computing persistence landscapes...")
    tda_features = []
    for idx, row in df.iterrows():
        point_cloud = row_to_pointcloud(row, numeric_cols)
        landscape_vec = compute_landscape(point_cloud)
        tda_features.append(landscape_vec)

    tda_features = np.array(tda_features)
    np.save(tda_output, tda_features)
    print(f"✔ TDA features saved to {tda_output} with shape {tda_features.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to CSV dataset"
    )
    parser.add_argument(
        "--tda_output", type=str, required=True, help="Path to save TDA features (.npy)"
    )
    parser.add_argument(
        "--corr_output",
        type=str,
        required=True,
        help="Path to save feature correlation (.npy)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.tda_output), exist_ok=True)
    run_tda(args.input_csv, args.tda_output, args.corr_output)
