"""Data cleaning, k-core filtering, and splitting."""

import pandas as pd

from .config import MIN_INTERACTIONS


def preprocess(
    data: pd.DataFrame,
    min_interactions: int = MIN_INTERACTIONS,
    positive_threshold: float | None = 4.0,
    max_iter: int = 5,
) -> tuple[pd.DataFrame, dict, dict, int, int]:
    original_len = len(data)
    if positive_threshold is not None and "rating" in data.columns:
        data = data[data["rating"] >= positive_threshold].copy()
        print(
            f"Positive feedback filter rating >= {positive_threshold}: "
            f"{original_len} -> {len(data)} interactions"
        )

    data = data.drop_duplicates(subset=["user_id", "item_id"], keep="last").reset_index(drop=True)
    dedup_len = len(data)
    for _ in range(max_iter):
        u_counts = data["user_id"].value_counts()
        i_counts = data["item_id"].value_counts()
        valid_users = u_counts[u_counts >= min_interactions].index
        valid_items = i_counts[i_counts >= min_interactions].index
        new_data = data[data["user_id"].isin(valid_users) & data["item_id"].isin(valid_items)]
        if len(new_data) == len(data):
            break
        data = new_data

    data = data.reset_index(drop=True)
    if len(data) != dedup_len:
        print(
            f"K-core filter min_interactions={min_interactions}: "
            f"{dedup_len} -> {len(data)} interactions"
        )

    user_ids = data["user_id"].unique()
    item_ids = data["item_id"].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {item: i for i, item in enumerate(item_ids)}
    data["u_idx"] = data["user_id"].map(user2idx)
    data["i_idx"] = data["item_id"].map(item2idx)
    return data, user2idx, item2idx, len(user_ids), len(item_ids)


def pairs_from_frame(data: pd.DataFrame) -> list[tuple[int, int]]:
    return list(zip(data["u_idx"].astype(int), data["i_idx"].astype(int)))


def global_timestamp_split(
    data: pd.DataFrame,
    t1: float,
    t2: float,
) -> tuple[pd.DataFrame, list[tuple[int, int]], list[tuple[int, int]]]:
    """Global absolute-timestamp split (Amazon Reviews 2023 benchmark 'timestamp' style).

    A SINGLE pair of cutoffs (t1, t2) is applied across all users at once:
        train = interactions with timestamp in (-inf, t1)
        valid = interactions with timestamp in [t1, t2)
        test  = interactions with timestamp in [t2, +inf)

    This is a per-DATASET absolute cut (not a per-user relative one): users whose
    entire history falls after t1 end up with zero train interactions (evaluated
    purely on their initial/untrained embedding) -- this is an intentional,
    realistic cold-start characteristic of this split style, not a bug, and
    matches how Amazon Reviews 2023's own 'timestamp' benchmark split behaves.
    """
    train_rows = data[data["timestamp"] < t1]
    valid_rows = data[(data["timestamp"] >= t1) & (data["timestamp"] < t2)]
    test_rows = data[data["timestamp"] >= t2]
    valid_pairs = pairs_from_frame(valid_rows)
    test_pairs = pairs_from_frame(test_rows)
    return train_rows.reset_index(drop=True), valid_pairs, test_pairs


def quantile_timestamp_cutoffs(
    data: pd.DataFrame,
    train_frac: float = 0.8405,
    valid_frac: float = 0.1034,
) -> tuple[float, float]:
    """Pick (t1, t2) so a dataset's OWN timestamp distribution splits into the
    same train/valid/test proportions as Amazon Reviews 2023's official
    'timestamp' split (0core All_Beauty: 84.05% / 10.34% / 5.61%), so the split
    style is comparable across datasets with very different absolute time ranges
    (e.g. Amazon ms-epoch up to 2023 vs MovieLens-1M s-epoch from 2000-2003).
    """
    ts = data["timestamp"].sort_values().to_numpy()
    t1 = ts[int(len(ts) * train_frac)]
    t2 = ts[int(len(ts) * (train_frac + valid_frac))]
    return float(t1), float(t2)
