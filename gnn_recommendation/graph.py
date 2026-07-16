"""Graph construction utilities."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def compute_edge_weight(
    train_df: pd.DataFrame,
    edge_mode: str = "binary",
    rating_scale: float = 5.0,
) -> np.ndarray | None:
    """Compute one weight per train interaction before graph symmetrization."""
    if edge_mode == "binary":
        return None
    if edge_mode == "rating":
        if "rating" not in train_df.columns:
            raise ValueError("edge_mode='rating' requires a rating column in train_df")
        weights = train_df["rating"].astype(float).to_numpy() / rating_scale
        return np.clip(weights, a_min=1e-8, a_max=None)
    if edge_mode == "time":
        if "timestamp" not in train_df.columns:
            raise ValueError("edge_mode='time' requires a timestamp column in train_df")
        timestamps = train_df["timestamp"].astype(float).to_numpy()
        t_min = np.nanmin(timestamps)
        t_max = np.nanmax(timestamps)
        span = max(t_max - t_min, 1.0)
        recency = (timestamps - t_min) / span
        return np.exp(-(1.0 - recency))
    raise ValueError(f"Unknown edge_mode: {edge_mode!r}. Expected binary, rating, or time.")


def build_normalized_graph(
    n_users: int,
    n_items: int,
    pairs: list[tuple[int, int]],
    device: torch.device,
    edge_weight: np.ndarray | list[float] | None = None,
    edge_mode: str = "binary",
) -> torch.Tensor:
    if edge_mode not in {"binary", "rating", "time"}:
        raise ValueError(f"Unknown edge_mode: {edge_mode!r}. Expected binary, rating, or time.")

    train_u = np.array([u for u, _ in pairs])
    train_i = np.array([i for _, i in pairs])
    if edge_weight is None:
        one_way_values = np.ones(len(pairs), dtype=np.float32)
    else:
        one_way_values = np.asarray(edge_weight, dtype=np.float32)
        if len(one_way_values) != len(pairs):
            raise ValueError(
                f"edge_weight length {len(one_way_values)} does not match pairs length {len(pairs)}"
            )
        if np.any(one_way_values < 0):
            raise ValueError("edge_weight must be non-negative")

    row = np.concatenate([train_u, train_i + n_users])
    col = np.concatenate([train_i + n_users, train_u])
    values = np.concatenate([one_way_values, one_way_values])

    n_nodes = n_users + n_items
    adjacency = sp.coo_matrix((values, (row, col)), shape=(n_nodes, n_nodes))
    degree = np.array(adjacency.sum(axis=1)).flatten()
    degree[degree == 0] = 1e-10
    d_inv_sqrt = np.power(degree, -0.5)
    normalized = (sp.diags(d_inv_sqrt) @ adjacency @ sp.diags(d_inv_sqrt)).tocoo()

    indices = torch.tensor(np.vstack([normalized.row, normalized.col]), dtype=torch.long)
    norm_values = torch.tensor(normalized.data, dtype=torch.float32)
    graph = torch.sparse_coo_tensor(indices, norm_values, size=(n_nodes, n_nodes))
    print(
        f"Graph: {n_nodes} nodes, {adjacency.nnz} directed edges after symmetrization "
        f"(edge_mode={edge_mode})"
    )
    return graph.coalesce().to(device)
