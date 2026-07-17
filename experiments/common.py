"""Shared helpers for M7 experiment scripts."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gnn_recommendation.pipeline import run_full_pipeline

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

# The 5 canonical graph-based models used for the report (see plan/README.md for
# their papers: LightGCN SIGIR'20, NGCF SIGIR'19, Sheaf4Rec ACM TORS'23/25,
# NCL WWW'22, DirectAU KDD'22). All 5 are faithful ports of official code.
FIVE_MODELS = ["LightGCN", "NGCF", "Sheaf4Rec-official", "NCL", "DirectAU"]


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "not-a-git-repo"


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dataset", default="amazon_beauty", help="Comma-separated dataset names.")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--no-gat", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Override to a fast smoke-test configuration.")
    parser.add_argument(
        "--models", default=",".join(FIVE_MODELS),
        help=f"Comma-separated model names to run. Default: the 5 canonical graph models ({', '.join(FIVE_MODELS)}).",
    )
    return parser


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def summarize_mean_std(raw: pd.DataFrame, metrics=("Recall@10", "NDCG@10")) -> pd.DataFrame:
    numeric = raw.select_dtypes(include="number").drop(
        columns=[col for col in ("seed",) if col in raw.columns],
        errors="ignore",
    )
    group_cols = [col for col in ["dataset", "model"] if col in raw.columns]
    config_cols = [
        col
        for col in [
            "n_layers",
            "latent_dim",
            "sheaf_stalk_dim",
            "restriction_type",
            "edge_mode",
            "positive_threshold",
            "epochs",
        ]
        if col in raw.columns
    ]
    group_cols.extend(config_cols)
    grouped = numeric.groupby([raw[col] for col in group_cols], dropna=False)
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std(ddof=0).add_suffix("_std")
    out = pd.concat([mean_df, std_df], axis=1).reset_index()
    priority = []
    for metric in metrics:
        for suffix in ("_mean", "_std"):
            col = metric + suffix
            if col in out.columns:
                priority.append(col)
    other_cols = [col for col in out.columns if col not in group_cols + priority]
    return out[group_cols + priority + other_cols]


def run_configs(
    configs: list[dict],
    output_csv: str,
    raw_output_csv: str | None = None,
    make_summary: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    ensure_dirs()
    commit = git_commit()
    frames = []
    for cfg in configs:
        datasets = cfg.pop("datasets")
        seeds = cfg.pop("seeds")
        metadata = cfg.pop("metadata", {})
        for dataset_name in datasets:
            for seed in seeds:
                print(f"\n=== {output_csv}: dataset={dataset_name} seed={seed} meta={metadata} ===")
                result = run_full_pipeline(dataset_name=dataset_name, seed=seed, **cfg).reset_index()
                result.insert(0, "dataset", dataset_name)
                result["seed"] = seed
                result["git_commit"] = commit
                for key, value in metadata.items():
                    result[key] = value
                frames.append(result)

    raw = pd.concat(frames, ignore_index=True)
    raw_path = Path(raw_output_csv or output_csv.replace(".csv", "_raw.csv"))
    raw.to_csv(raw_path, index=False)
    if make_summary:
        summary = summarize_mean_std(raw)
        summary.to_csv(output_csv, index=False)
    else:
        summary = None
        raw.to_csv(output_csv, index=False)
    print(f"Wrote raw results to {raw_path}")
    if summary is not None:
        print(f"Wrote summary results to {output_csv}")
    return raw, summary


def save_bar(df: pd.DataFrame, x: str, y: str, output: str, hue: str | None = None, title: str | None = None):
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if hue and hue in df.columns:
        pivot = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
        pivot.plot(kind="bar", ax=ax)
    else:
        df.plot(kind="bar", x=x, y=y, ax=ax, legend=False)
    ax.set_title(title or y)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_line(df: pd.DataFrame, x: str, y: str, output: str, hue: str | None = None, title: str | None = None):
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if hue and hue in df.columns:
        for key, sub in df.groupby(hue):
            sub = sub.sort_values(x)
            ax.plot(sub[x], sub[y], marker="o", label=str(key))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    else:
        df = df.sort_values(x)
        ax.plot(df[x], df[y], marker="o")
    ax.set_title(title or y)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
