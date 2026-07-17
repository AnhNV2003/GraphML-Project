"""Run M2 exploratory data analysis on local real datasets."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import pandas as pd

from gnn_recommendation.data import (
    load_amazon_category_benchmark,
    load_amazon_category_raw,
    load_movielens_1m_real,
)
from gnn_recommendation.eda import (
    kcore_comparison,
    save_eda_figures,
    summarize_dataset,
    write_markdown_summary,
)


# Nguồn dữ liệu raw cho EDA. slug -> (loader, timestamp_unit).
RAW_SOURCES = {
    "amazon_video_games_raw": (lambda: load_amazon_category_raw("Video_Games"), "ms"),
    "movielens_1m": (load_movielens_1m_real, "s"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run M2 EDA on local recommendation datasets.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument(
        "--datasets",
        default="amazon_video_games_raw,movielens_1m",
        help=(
            "Comma-separated raw dataset slugs to run EDA on. "
            f"Choices: {', '.join(RAW_SOURCES)}."
        ),
    )
    parser.add_argument(
        "--benchmark-category",
        default=None,
        help=(
            "Amazon category name to also summarize from the bundled 0core/5core "
            "timestamp benchmark (e.g. Video_Games). Omit to skip benchmark comparison."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    requested = [s.strip() for s in args.datasets.split(",") if s.strip()]
    unknown = [s for s in requested if s not in RAW_SOURCES]
    if unknown:
        raise SystemExit(
            f"Unknown dataset(s): {unknown}. Choices: {', '.join(RAW_SOURCES)}"
        )
    datasets = {}
    for slug in requested:
        loader, unit = RAW_SOURCES[slug]
        datasets[slug] = {"data": loader(), "timestamp_unit": unit}

    benchmark_variants = {}
    if args.benchmark_category:
        cat = args.benchmark_category
        benchmark_variants = {
            f"{cat.lower()}_benchmark_0core_timestamp": load_amazon_category_benchmark(
                cat, core="0core", split="timestamp",
            ),
            f"{cat.lower()}_benchmark_5core_timestamp": load_amazon_category_benchmark(
                cat, core="5core", split="timestamp",
            ),
        }

    dataset_rows = []
    kcore_rows = []
    for name, cfg in datasets.items():
        data = cfg["data"]
        dataset_rows.append(summarize_dataset(data, name))
        kcore_rows.extend(kcore_comparison(data, name, min_interactions=args.min_interactions))
        save_eda_figures(data, name, figures_dir, timestamp_unit=cfg["timestamp_unit"])

    benchmark_rows = []
    for name, data in benchmark_variants.items():
        benchmark_rows.append(summarize_dataset(data, name))

    dataset_stats = pd.DataFrame(dataset_rows)
    kcore_stats = pd.DataFrame(kcore_rows)
    benchmark_stats = pd.DataFrame(benchmark_rows)

    dataset_stats.to_csv(tables_dir / "dataset_statistics.csv", index=False)
    kcore_stats.to_csv(tables_dir / "kcore_comparison.csv", index=False)
    benchmark_stats.to_csv(tables_dir / "amazon_benchmark_comparison.csv", index=False)

    data_choice = (
        "Amazon Video Games raw is the preferred source for the main modelling pipeline after M3 "
        "filtering because it preserves the full local dataset before rating-threshold and k-core "
        "choices. The bundled 5-core benchmark is useful for smoke tests and reference-aligned "
        "experiments, but it is very small compared with raw/0-core data. MovieLens uses the local "
        "ratings.dat file as the canonical interaction source."
    )
    write_markdown_summary(
        output_dir / "M2_eda_summary.md",
        pd.concat([dataset_stats, benchmark_stats], ignore_index=True),
        kcore_stats,
        data_choice,
    )

    print("Wrote EDA figures to", figures_dir)
    print("Wrote EDA tables to", tables_dir)
    print("Wrote summary to", output_dir / "M2_eda_summary.md")


if __name__ == "__main__":
    main()
