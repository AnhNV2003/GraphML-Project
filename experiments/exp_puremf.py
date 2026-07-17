"""Train & evaluate PureMF (official LightGCN-PyTorch model) as an extra baseline.

PureMF is imported directly from the official gusye1234/LightGCN-PyTorch repo
(no reimplementation) via `include_auxiliary=True` in build_all_models.

Usage:
  python -m experiments.exp_puremf --dataset amazon_video_games,movielens_1m
"""

import argparse

from experiments.common import add_common_args, comma_list, int_list, run_configs


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--output", default="results/puremf_comparison.csv")
    args = parser.parse_args()

    datasets = comma_list(args.dataset)
    seeds = [42] if args.quick else int_list(args.seeds)
    epochs = 2 if args.quick else args.epochs

    configs = [
        {
            "datasets": datasets,
            "seeds": seeds,
            "epochs": epochs,
            "patience": args.patience,
            "include_gat": False,
            "include_auxiliary": True,
            "model_names": ["PureMF"],
            "metadata": {"epochs": epochs},
        }
    ]
    run_configs(configs, args.output)


if __name__ == "__main__":
    main()
