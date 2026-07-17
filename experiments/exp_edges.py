"""E4: edge-construction (weighting) study — binary / rating / time.

Usage:
  python -m experiments.exp_edges --dataset amazon_video_games --edge-modes binary,rating,time
  python -m experiments.exp_edges --models LightGCN,Sheaf4Rec-official
"""

import argparse

from experiments.common import add_common_args, comma_list, int_list, run_configs


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--edge-modes", default="binary,rating,time", help="Comma-separated edge weighting modes.")
    parser.add_argument("--output", default="results/edge_construction.csv")
    args = parser.parse_args()

    datasets = comma_list(args.dataset)
    seeds = [42] if args.quick else int_list(args.seeds)
    epochs = 1 if args.quick else args.epochs
    edge_modes = comma_list(args.edge_modes)
    model_names = comma_list(args.models)

    configs = [
        {
            "datasets": datasets,
            "seeds": seeds,
            "epochs": epochs,
            "patience": args.patience,
            "edge_mode": edge_mode,
            "include_gat": False,
            "include_auxiliary": False,
            "model_names": model_names,
            "metadata": {"epochs": epochs, "edge_mode": edge_mode},
        }
        for edge_mode in edge_modes
    ]
    run_configs(configs, args.output)


if __name__ == "__main__":
    main()
