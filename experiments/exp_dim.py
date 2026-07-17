"""E2: embedding-dimension sweep across the 5 canonical graph models.

For Sheaf4Rec-official, also sweeps its stalk dimension (sheaf_stalk_dim)
separately, since that is an architecture-specific knob on top of latent_dim.

Usage:
  python -m experiments.exp_dim --dataset amazon_video_games --latent-dims 16,32,64,128
  python -m experiments.exp_dim --models LightGCN,NGCF --sheaf-dims ""   # skip sheaf-dim sweep
"""

import argparse

from experiments.common import add_common_args, comma_list, int_list, run_configs


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--latent-dims", default="16,32,64,128", help="Comma-separated latent (embedding) dims.")
    parser.add_argument(
        "--sheaf-dims", default="1,2,3,4,6,8",
        help="Comma-separated sheaf stalk dims (Sheaf4Rec-official only; empty string to skip).",
    )
    parser.add_argument("--output", default="results/dim_sweep.csv")
    args = parser.parse_args()

    datasets = comma_list(args.dataset)
    seeds = [42] if args.quick else int_list(args.seeds)
    epochs = 1 if args.quick else args.epochs
    latent_dims = [64] if args.quick else int_list(args.latent_dims)
    sheaf_dims = [] if args.quick or not args.sheaf_dims.strip() else int_list(args.sheaf_dims)
    model_names = comma_list(args.models)

    configs = []
    for latent_dim in latent_dims:
        configs.append(
            {
                "datasets": datasets,
                "seeds": seeds,
                "epochs": epochs,
                "embed_dim": latent_dim,
                "patience": args.patience,
                "include_gat": False,
                "include_auxiliary": False,
                "model_names": model_names,
                "metadata": {"epochs": epochs, "latent_dim": latent_dim},
            }
        )
    if sheaf_dims and "Sheaf4Rec-official" in model_names:
        for sheaf_dim in sheaf_dims:
            configs.append(
                {
                    "datasets": datasets,
                    "seeds": seeds,
                    "epochs": epochs,
                    "patience": args.patience,
                    "include_gat": False,
                    "include_auxiliary": False,
                    "model_names": ["Sheaf4Rec-official"],
                    "config_overrides": {"sheaf_stalk_dim": sheaf_dim},
                    "metadata": {"epochs": epochs, "sheaf_stalk_dim": sheaf_dim},
                }
            )
    run_configs(configs, args.output)


if __name__ == "__main__":
    main()
