"""Plotting helpers for result tables.

Note: figure functions here return the Axes instead of calling plt.show(),
since every entry point in this repo runs with the headless "Agg" backend
(set in experiments/common.py, build_report.py, run_pipeline.py, run_eda.py).
Call fig.savefig(...) or plt.show() yourself if using this interactively.
"""

import matplotlib.pyplot as plt

from .data import DATASET_REGISTRY


def plot_metric_by_dataset(combined_results, metric: str = "NDCG@10"):
    pivot = combined_results[metric].unstack("dataset")
    pivot = pivot.rename(columns={k: v["display_name"] for k, v in DATASET_REGISTRY.items()})
    ax = pivot.plot(kind="bar", figsize=(11, 5), colormap="viridis")
    ax.set_title(f"{metric} by model and dataset")
    ax.set_ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Dataset")
    plt.tight_layout()
    return ax


def print_best_models(combined_results, metric: str = "NDCG@10") -> None:
    print(f"Best model by {metric}, per dataset:")
    for dataset_name in combined_results.index.get_level_values("dataset").unique():
        sub = combined_results.loc[dataset_name]
        best = sub[metric].idxmax()
        display_name = DATASET_REGISTRY.get(dataset_name, {}).get("display_name", dataset_name)
        print(f"  {display_name}: {best} ({metric} = {sub.loc[best, metric]:.4f})")

