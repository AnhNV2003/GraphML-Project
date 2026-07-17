"""Build the final comparison tables + figures from results/*.csv.

Reads the summary CSVs written by train.sh (E0-E4, SOTA, TAG-CF, PureMF) and
produces:
  - results/tables/model_comparison_<dataset>.md  (markdown table, ranked by NDCG@10)
  - results/figures/model_comparison_<dataset>.png (bar chart, Recall@10 & NDCG@10)
  - results/figures/e1_layer_sweep.png, e2_dim_sweep.png, e3_expressiveness.png,
    e4_edge_construction.png (whichever inputs are present)
  - results/M8_comparison.md (combined report across all datasets)

Usage:
  python -m experiments.build_report
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

MEDALS = ["🥇", "🥈", "🥉"]


def _rank_medal(i: int) -> str:
    return MEDALS[i] if i < len(MEDALS) else str(i + 1)


def _mean_col(df: pd.DataFrame, metric: str) -> str:
    return f"{metric}_mean" if f"{metric}_mean" in df.columns else metric


def _normalize_metric_columns(df: pd.DataFrame, metrics=("Recall@10", "NDCG@10")) -> pd.DataFrame:
    """Some experiment scripts (exp_tagcf.py) write raw `Recall@10`/`NDCG@10`
    columns directly (no run_configs/summarize_mean_std pass), while others
    write `Recall@10_mean`/`NDCG@10_mean`. Normalize every frame to the plain
    metric name before concatenating so rows don't fragment into all-NaN pairs."""
    df = df.copy()
    for metric in metrics:
        mean_col = f"{metric}_mean"
        if mean_col in df.columns:
            if metric in df.columns:
                df[metric] = df[metric].fillna(df[mean_col])
            else:
                df[metric] = df[mean_col]
    return df


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[skip] {path} not found")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"[skip] {path} is empty")
        return None
    return df


def build_model_comparison():
    """Merge main_comparison (E0), sota_comparison, tagcf_comparison, puremf_comparison
    into one per-dataset table ranked by NDCG@10."""
    frames = []
    for path in [
        RESULTS_DIR / "main_comparison.csv",
        RESULTS_DIR / "sota_comparison.csv",
        RESULTS_DIR / "tagcf_comparison.csv",
        RESULTS_DIR / "puremf_comparison.csv",
    ]:
        df = load_csv(path)
        if df is not None:
            frames.append(_normalize_metric_columns(df))
    if not frames:
        print("[build_model_comparison] no input CSVs found, skipping.")
        return

    combined = pd.concat(frames, ignore_index=True)
    recall_col, ndcg_col = "Recall@10", "NDCG@10"
    if recall_col not in combined.columns or ndcg_col not in combined.columns:
        print("[build_model_comparison] Recall@10/NDCG@10 columns missing, skipping.")
        return

    # Keep only the best row per (dataset, model): a model can appear in several
    # source CSVs (e.g. DirectAU/NCL in both main + sota) or across raw seeds
    # (MF/MF+TAG-CF). Rank by NDCG@10 and drop duplicates so each model shows once.
    combined = (
        combined.sort_values(ndcg_col, ascending=False)
        .drop_duplicates(subset=["dataset", "model"], keep="first")
        .reset_index(drop=True)
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report_sections = ["# Model Comparison (M8)\n\n_Global timestamp split (Amazon Reviews 2023 benchmark style); full-catalog ranking._\n"]

    for dataset, group in combined.groupby("dataset"):
        ranked = group.sort_values(ndcg_col, ascending=False).reset_index(drop=True)

        # Markdown table
        lines = [f"\n### {dataset}\n", "| Rank | Model | Recall@10 | NDCG@10 |", "|---|---|---|---|"]
        for i, row in ranked.iterrows():
            lines.append(f"| {_rank_medal(i)} | {row['model']} | {row[recall_col]:.4f} | {row[ndcg_col]:.4f} |")
        table_md = "\n".join(lines)
        (TABLES_DIR / f"model_comparison_{dataset}.md").write_text(table_md + "\n")
        report_sections.append(table_md)

        # Bar chart: Recall@10 & NDCG@10 side by side
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(ranked))
        width = 0.35
        ax.bar([i - width / 2 for i in x], ranked[recall_col], width, label="Recall@10")
        ax.bar([i + width / 2 for i in x], ranked[ndcg_col], width, label="NDCG@10")
        ax.set_xticks(list(x))
        ax.set_xticklabels(ranked["model"], rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"Model comparison — {dataset}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"model_comparison_{dataset}.png", dpi=160)
        plt.close(fig)
        print(f"[build_model_comparison] wrote table + figure for {dataset}")

    (RESULTS_DIR / "M8_comparison.md").write_text("\n".join(report_sections) + "\n")
    print(f"[build_model_comparison] wrote {RESULTS_DIR / 'M8_comparison.md'}")


def build_sweep_figure(csv_name: str, x_col: str, out_name: str, title: str, hue_col: str = "model"):
    df = load_csv(RESULTS_DIR / csv_name)
    if df is None:
        return
    ndcg_col = _mean_col(df, "NDCG@10")
    if x_col not in df.columns or ndcg_col not in df.columns:
        print(f"[{out_name}] missing columns ({x_col}/{ndcg_col}), skipping.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    group_col = hue_col if hue_col in df.columns else None
    if group_col:
        for key, sub in df.groupby(group_col):
            sub = sub.dropna(subset=[x_col]).sort_values(x_col)
            if sub.empty:
                continue
            ax.plot(sub[x_col], sub[ndcg_col], marker="o", label=str(key))
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    else:
        sub = df.dropna(subset=[x_col]).sort_values(x_col)
        ax.plot(sub[x_col], sub[ndcg_col], marker="o")
    ax.set_xlabel(x_col)
    ax.set_ylabel("NDCG@10")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, dpi=160)
    plt.close(fig)
    print(f"[{out_name}] wrote figure from {csv_name}")


def main():
    build_model_comparison()
    build_sweep_figure("layer_sweep.csv", "n_layers", "e1_layer_sweep.png", "E1: Layer-count sweep")
    build_sweep_figure("dim_sweep.csv", "latent_dim", "e2_dim_sweep_latent.png", "E2: Latent-dim sweep")
    build_sweep_figure("dim_sweep.csv", "sheaf_stalk_dim", "e2_dim_sweep_sheaf.png", "E2: Sheaf stalk-dim sweep (Sheaf4Rec-official)")
    build_sweep_figure("expressiveness.csv", "model", "e3_expressiveness.png", "E3: Restriction-map expressiveness (Sheaf4Rec)", hue_col=None)
    build_sweep_figure("edge_construction.csv", "edge_mode", "e4_edge_construction.png", "E4: Edge-construction study")
    print("\nDone. See results/M8_comparison.md, results/tables/, results/figures/")


if __name__ == "__main__":
    main()
