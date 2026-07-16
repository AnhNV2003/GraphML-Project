"""Hyperparameter tuning for the official Sheaf4Rec (Purificato et al.).

Sweeps the most impactful knobs (n_layers, latent_dim, lr) and selects the best
configuration by VALIDATION NDCG@10 (M6 protocol), then reports test metrics.
LightGCN is run once as a reference baseline.

Usage:
  python -m experiments.exp_tune_official --dataset movielens_1m --seeds 42 --epochs 100 --patience 10
"""

import argparse
from pathlib import Path

import pandas as pd

from experiments.common import add_common_args, comma_list, ensure_dirs, git_commit, int_list
from gnn_recommendation.pipeline import run_full_pipeline


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--layers", default="3,4,5", help="n_layers grid")
    parser.add_argument("--latent-dims", default="64,128", help="latent_dim grid")
    parser.add_argument("--lrs", default="0.001", help="learning-rate grid")
    parser.add_argument("--output", default="results/tune_official.csv")
    parser.add_argument("--with-lightgcn", action="store_true", help="Also run LightGCN as a reference")
    args = parser.parse_args()

    dataset = comma_list(args.dataset)[0]
    seed = (int_list(args.seeds) or [42])[0]
    epochs = 1 if args.quick else args.epochs
    layers = [3] if args.quick else int_list(args.layers)
    latent_dims = [64] if args.quick else int_list(args.latent_dims)
    lrs = [0.001] if args.quick else [float(x) for x in comma_list(args.lrs)]

    ensure_dirs()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    rows = []

    # Reference baseline (optional) — LightGCN at default config.
    if args.with_lightgcn:
        print("\n=== reference: LightGCN ===")
        res = run_full_pipeline(
            dataset_name=dataset, seed=seed, epochs=epochs, patience=args.patience,
            include_gat=False, include_auxiliary=False, model_names=["LightGCN"],
        ).reset_index()
        res["config"] = "lightgcn/default"
        rows.append(res)

    for n_layers in layers:
        for latent in latent_dims:
            for lr in lrs:
                tag = f"L{n_layers}_d{latent}_lr{lr}"
                print(f"\n=== official: {tag} ===")
                res = run_full_pipeline(
                    dataset_name=dataset,
                    seed=seed,
                    embed_dim=latent,
                    epochs=epochs,
                    patience=args.patience,
                    include_gat=False,
                    include_auxiliary=False,
                    model_names=["Sheaf4Rec-official"],
                    config_overrides={"sheaf_n_layers": n_layers, "lr": lr},
                ).reset_index()
                res["config"] = tag
                res["n_layers"] = n_layers
                res["latent_dim"] = latent
                res["lr"] = lr
                rows.append(res)

    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "dataset", dataset)
    out["seed"] = seed
    out["git_commit"] = commit
    out.to_csv(args.output, index=False)
    print(f"\nWrote tuning results to {args.output}")

    # Rank official configs by validation NDCG@10 (model selection).
    off = out[out["model"] == "Sheaf4Rec-official"].copy()
    val_col = "val_NDCG@10" if "val_NDCG@10" in off.columns else "best_val_NDCG@10"
    ranked = off.sort_values(val_col, ascending=False)
    show = [c for c in ["config", val_col, "NDCG@10", "Recall@10", "train_seconds"] if c in ranked.columns]
    print("\n=== official configs ranked by validation NDCG@10 ===")
    print(ranked[show].to_string(index=False))
    if not ranked.empty:
        best = ranked.iloc[0]
        print(f"\nBEST by val: {best['config']}  |  val={best[val_col]:.4f}  test NDCG@10={best['NDCG@10']:.4f}")


if __name__ == "__main__":
    main()
