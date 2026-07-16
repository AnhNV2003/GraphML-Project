"""Command-line entry point for the extracted recommendation pipeline."""

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

from gnn_recommendation.config import EMBED_DIM, MIN_INTERACTIONS, N_EPOCHS_MULTI, N_LAYERS, set_global_seed
from gnn_recommendation.data import DATASET_REGISTRY
from gnn_recommendation.pipeline import run_multi_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run GNN recommendation experiments.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_REGISTRY),
        help="Dataset to run. Repeat for multiple datasets. Defaults to all datasets.",
    )
    parser.add_argument("--epochs", type=int, default=N_EPOCHS_MULTI)
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds, e.g. 42,43,44.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience on val NDCG@10.")
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--min-interactions", type=int, default=MIN_INTERACTIONS)
    parser.add_argument(
        "--positive-threshold",
        default="4.0",
        help="Implicit-positive rating threshold. Use 'none' to keep all ratings.",
    )
    parser.add_argument(
        "--edge-mode",
        choices=["binary", "rating", "time"],
        default="binary",
        help="Graph edge weighting mode.",
    )
    parser.add_argument("--output", default="multi_dataset_results.csv")
    parser.add_argument("--repo-dir", default="LightGCN-PyTorch")
    parser.add_argument("--no-gat", action="store_true", help="Skip GAT if torch-geometric is unavailable.")
    parser.add_argument(
        "--main-models-only",
        action="store_true",
        help="Skip auxiliary models such as PureMF/UltraGCN where supported.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names to run, e.g. Popularity,LightGCN,Sheaf4Rec-full_sheaf.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    set_global_seed(seeds[0])
    positive_threshold = None if args.positive_threshold.lower() == "none" else float(args.positive_threshold)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()] if args.models else None
    results = run_multi_dataset(
        dataset_names=args.dataset,
        output_csv=args.output,
        seeds=seeds,
        epochs=args.epochs,
        embed_dim=args.embed_dim,
        n_layers=args.layers,
        min_interactions=args.min_interactions,
        positive_threshold=positive_threshold,
        edge_mode=args.edge_mode,
        include_gat=not args.no_gat,
        include_auxiliary=not args.main_models_only,
        repo_dir=args.repo_dir,
        patience=args.patience,
        model_names=model_names,
    )
    print(results.round(4))
    if len(seeds) == 1 and "NDCG@10" in results.columns:
        from gnn_recommendation.plots import print_best_models

        print_best_models(results)


if __name__ == "__main__":
    main()
