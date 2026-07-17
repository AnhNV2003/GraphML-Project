"""Train & evaluate the Popularity baseline (non-personalized top-popular items).

Popularity is a non-trainable baseline: it just counts item frequencies over the
train interactions and recommends the globally most popular items to every user
(see gnn_recommendation/extra_models.py:PopularityRec). There is no gradient step,
so "training" is effectively instant; the cost is dataset load + preprocessing +
the ranking evaluation. It is the standard lower-bound reference for whether a
personalized model learns anything beyond global popularity.

Usage:
  python -m experiments.exp_popularity --dataset amazon_video_games --seeds 42,43,44
"""

import argparse

from experiments.common import add_common_args, comma_list, int_list, run_configs


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--output", default="results/popularity_comparison.csv")
    args = parser.parse_args()

    datasets = comma_list(args.dataset)
    seeds = [42] if args.quick else int_list(args.seeds)
    # Popularity has no epochs; epochs are ignored for the (non-trainable) model
    # but kept in metadata for a uniform schema with the other comparison CSVs.
    epochs = 2 if args.quick else args.epochs

    configs = [
        {
            "datasets": datasets,
            "seeds": seeds,
            "epochs": epochs,
            "patience": args.patience,
            "include_gat": False,
            "include_auxiliary": True,
            "model_names": ["Popularity"],
            "metadata": {"epochs": epochs},
        }
    ]
    run_configs(configs, args.output)


if __name__ == "__main__":
    main()
