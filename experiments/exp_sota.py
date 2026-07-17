"""Train & evaluate the 5 ported SOTA self-supervised graph-CF models sequentially
on both datasets, using the same protocol (epochs/patience/seeds) as E0.

Models (see gnn_recommendation/ssl_models.py for faithful-port details + sources):
  SGL (SIGIR'21), SimGCL (SIGIR'22), DirectAU (KDD'22), NCL (WWW'22), LightGCL (ICLR'23)

Usage:
  python -m experiments.exp_sota --dataset amazon_video_games,movielens_1m --seeds 42 --epochs 100 --patience 10
"""

import argparse

from experiments.common import add_common_args, comma_list, int_list, run_configs


def main():
    parser = add_common_args(argparse.ArgumentParser(conflict_handler="resolve"))
    parser.add_argument(
        "--models", default="SGL,SimGCL,DirectAU,NCL,LightGCL",
        help="Comma-separated subset of the 5 SOTA self-supervised models to run.",
    )
    parser.add_argument("--output", default="results/sota_comparison.csv")
    args = parser.parse_args()

    datasets = comma_list(args.dataset)
    seeds = [42] if args.quick else int_list(args.seeds)
    epochs = 2 if args.quick else args.epochs
    model_names = comma_list(args.models)

    configs = [
        {
            "datasets": datasets,
            "seeds": seeds,
            "epochs": epochs,
            "patience": args.patience,
            "include_gat": False,
            "include_auxiliary": False,
            "model_names": model_names,
            "metadata": {"epochs": epochs},
        }
    ]
    run_configs(configs, args.output)


if __name__ == "__main__":
    main()
