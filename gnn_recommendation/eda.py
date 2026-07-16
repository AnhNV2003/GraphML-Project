"""Exploratory summaries and plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import MIN_INTERACTIONS
from .preprocessing import preprocess


def summarize_interactions(data: pd.DataFrame) -> dict[str, float]:
    n_users = data["user_id"].nunique()
    n_items = data["item_id"].nunique()
    n_interactions = len(data)
    sparsity = 1 - n_interactions / (n_users * n_items)
    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "sparsity": sparsity,
    }


def rating_summary(data: pd.DataFrame) -> dict[str, float]:
    n_interactions = len(data)
    positives = int((data["rating"] >= 4).sum())
    return {
        "rating_min": float(data["rating"].min()),
        "rating_max": float(data["rating"].max()),
        "rating_mean": float(data["rating"].mean()),
        "positive_rating_ge_4": positives,
        "positive_rating_ge_4_ratio": positives / n_interactions if n_interactions else 0.0,
    }


def history_length_summary(data: pd.DataFrame, key: str) -> dict[str, float]:
    counts = data[key].value_counts()
    quantiles = counts.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        f"{key}_history_min": int(counts.min()),
        f"{key}_history_mean": float(counts.mean()),
        f"{key}_history_median": float(quantiles.loc[0.5]),
        f"{key}_history_p90": float(quantiles.loc[0.9]),
        f"{key}_history_p95": float(quantiles.loc[0.95]),
        f"{key}_history_p99": float(quantiles.loc[0.99]),
        f"{key}_history_max": int(counts.max()),
    }


def summarize_dataset(data: pd.DataFrame, name: str) -> dict[str, float | str]:
    summary: dict[str, float | str] = {"dataset": name}
    summary.update(summarize_interactions(data))
    summary.update(rating_summary(data))
    summary.update(history_length_summary(data, "user_id"))
    summary.update(history_length_summary(data, "item_id"))
    return summary


def kcore_comparison(
    data: pd.DataFrame,
    name: str,
    min_interactions: int = MIN_INTERACTIONS,
    positive_threshold: float | None = None,
) -> list[dict[str, float | str]]:
    raw_summary = summarize_interactions(data)
    filtered, _, _, _, _ = preprocess(
        data,
        min_interactions=min_interactions,
        positive_threshold=positive_threshold,
    )
    filtered_summary = summarize_interactions(filtered)
    return [
        {
            "dataset": name,
            "stage": "before_kcore",
            "min_interactions": min_interactions,
            **raw_summary,
        },
        {
            "dataset": name,
            "stage": "after_kcore",
            "min_interactions": min_interactions,
            **filtered_summary,
        },
    ]


def _timestamp_series(data: pd.DataFrame, unit: str) -> pd.Series:
    return pd.to_datetime(data["timestamp"], unit=unit, errors="coerce")


def save_eda_figures(
    data: pd.DataFrame,
    dataset_slug: str,
    output_dir: Path,
    timestamp_unit: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    user_counts = data["user_id"].value_counts()
    item_counts = data["item_id"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(user_counts.values, bins=50, color="darkorange")
    axes[0].set_title(f"{dataset_slug}: interactions per user")
    axes[0].set_xlabel("Interactions")
    axes[0].set_ylabel("Users")
    axes[0].set_yscale("log")
    axes[1].hist(item_counts.values, bins=50, color="seagreen")
    axes[1].set_title(f"{dataset_slug}: interactions per item")
    axes[1].set_xlabel("Interactions")
    axes[1].set_ylabel("Items")
    axes[1].set_yscale("log")
    fig.tight_layout()
    path = output_dir / f"{dataset_slug}_history_lengths.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved_paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    data["rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(f"{dataset_slug}: rating distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Interactions")
    fig.tight_layout()
    path = output_dir / f"{dataset_slug}_rating_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved_paths.append(path)

    timestamps = _timestamp_series(data, timestamp_unit).dropna()
    monthly = timestamps.dt.to_period("M").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index.astype(str), monthly.values, color="purple")
    ax.set_title(f"{dataset_slug}: interactions over time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Interactions")
    step = max(len(monthly) // 10, 1)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels(monthly.index.astype(str)[::step], rotation=45, ha="right")
    fig.tight_layout()
    path = output_dir / f"{dataset_slug}_interactions_over_time.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved_paths.append(path)

    return saved_paths


def write_markdown_summary(
    output_path: Path,
    dataset_stats: pd.DataFrame,
    kcore_stats: pd.DataFrame,
    data_choice: str,
) -> None:
    def to_markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        formatted = df.copy()
        for col in formatted.columns:
            if pd.api.types.is_float_dtype(formatted[col]):
                formatted[col] = formatted[col].map(lambda value: f"{value:.6f}")
            else:
                formatted[col] = formatted[col].map(str)
        header = "| " + " | ".join(formatted.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
        rows = [
            "| " + " | ".join(row) + " |"
            for row in formatted.astype(str).itertuples(index=False, name=None)
        ]
        return "\n".join([header, separator, *rows])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        "# M2 EDA Summary",
        "",
        "## Dataset Statistics",
        "",
        to_markdown_table(dataset_stats),
        "",
        "## Before/After K-Core",
        "",
        to_markdown_table(kcore_stats),
        "",
        "## Data Configuration Decision",
        "",
        data_choice,
        "",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def eda_plots(data: pd.DataFrame, name: str = "dataset") -> dict[str, float]:
    summary = summarize_interactions(data)
    print(f"[{name}] Users: {summary['n_users']}")
    print(f"[{name}] Items: {summary['n_items']}")
    print(f"[{name}] Interactions: {summary['n_interactions']}")
    print(f"[{name}] Sparsity: {summary['sparsity']:.4%}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    data["rating"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title(f"[{name}] Rating distribution")
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Count")

    user_counts = data["user_id"].value_counts()
    axes[1].hist(user_counts.values, bins=50, color="darkorange")
    axes[1].set_title(f"[{name}] Interactions per user")
    axes[1].set_xlabel("Interactions")
    axes[1].set_ylabel("Users")
    axes[1].set_yscale("log")

    item_counts = data["item_id"].value_counts().sort_values(ascending=False).values
    axes[2].plot(np.arange(len(item_counts)), item_counts, color="seagreen")
    axes[2].set_title(f"[{name}] Long-tail item popularity")
    axes[2].set_xlabel("Item rank")
    axes[2].set_ylabel("Interactions")

    plt.tight_layout()
    plt.show()
    return summary
