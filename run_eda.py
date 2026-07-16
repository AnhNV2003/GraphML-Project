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


def parse_args():
    parser = argparse.ArgumentParser(description="Run M2 EDA on local recommendation datasets.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--min-interactions", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "amazon_beauty_raw": {
            "data": load_amazon_category_raw("All_Beauty"),
            "timestamp_unit": "ms",
        },
        "movielens_1m": {
            "data": load_movielens_1m_real(),
            "timestamp_unit": "s",
        },
    }

    benchmark_variants = {
        "amazon_beauty_benchmark_0core_timestamp": load_amazon_category_benchmark(
            "All_Beauty",
            core="0core",
            split="timestamp",
        ),
        "amazon_beauty_benchmark_5core_timestamp": load_amazon_category_benchmark(
            "All_Beauty",
            core="5core",
            split="timestamp",
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
        "Amazon All Beauty raw is the preferred source for the main modelling pipeline after M3 "
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
