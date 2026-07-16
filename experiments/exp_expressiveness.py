"""E3: restriction-map expressiveness study (Sheaf4Rec-official only).

gcn_like = fixed identity restriction (0 learned params)
gat_like = scalar-attention restriction (81 params)
full_sheaf = full d x d learned restriction maps (216 params)

This concept (restriction map) is specific to the sheaf-diffusion architecture
and does not apply to LightGCN/NGCF/NCL/DirectAU, so this experiment is scoped
to Sheaf4Rec variants only (no --models tag).

Usage:
  python -m experiments.exp_expressiveness --dataset amazon_beauty --seeds 42,43,44 --epochs 50 --patience 10
"""

import argparse

from experiments.common import add_common_args, comma_list, int_list, run_configs


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--output", default="results/expressiveness.csv")
    args = parser.parse_args()

    datasets = comma_list(args.dataset)
    seeds = [42] if args.quick else int_list(args.seeds)
    epochs = 1 if args.quick else args.epochs
    model_names = ["Sheaf4Rec-gcn_like", "Sheaf4Rec-gat_like", "Sheaf4Rec-full_sheaf"]

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
