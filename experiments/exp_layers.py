"""E1: layer-count sweep (n_layers) across the 5 canonical graph models.

Usage:
  python -m experiments.exp_layers --dataset amazon_beauty --layers 1,2,3,4,5,6,8
  python -m experiments.exp_layers --models Sheaf4Rec-official,LightGCN
"""

import argparse

from experiments.common import add_common_args, comma_list, int_list, run_configs


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--layers", default="1,2,3,4,5,6,8", help="Comma-separated n_layers values.")
    parser.add_argument("--output", default="results/layer_sweep.csv")
    args = parser.parse_args()

    datasets = comma_list(args.dataset)
    seeds = [42] if args.quick else int_list(args.seeds)
    epochs = 1 if args.quick else args.epochs
    layers = [3] if args.quick else int_list(args.layers)
    model_names = comma_list(args.models)

    configs = [
        {
            "datasets": datasets,
            "seeds": seeds,
            "epochs": epochs,
            "patience": args.patience,
            "n_layers": n_layers,
            "include_gat": False,
            "include_auxiliary": False,
            "model_names": model_names,
            "metadata": {"epochs": epochs, "n_layers": n_layers},
        }
        for n_layers in layers
    ]
    run_configs(configs, args.output)


if __name__ == "__main__":
    main()
