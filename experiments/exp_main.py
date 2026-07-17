"""E0: main model comparison table (5 canonical graph models).

Usage:
  python -m experiments.exp_main --dataset amazon_video_games,movielens_1m --seeds 42,43,44 --epochs 100 --patience 10
  python -m experiments.exp_main --models LightGCN,NGCF   # subset only
"""

import argparse

from experiments.common import add_common_args, comma_list, int_list, run_configs


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--output", default="results/main_comparison.csv")
    args = parser.parse_args()

    datasets = comma_list(args.dataset)
    seeds = [42] if args.quick else int_list(args.seeds)
    epochs = 1 if args.quick else args.epochs
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
            "metadata": {"epochs": epochs, "edge_mode": "binary", "positive_threshold": 4.0},
        }
    ]
    run_configs(configs, args.output)


if __name__ == "__main__":
    main()
