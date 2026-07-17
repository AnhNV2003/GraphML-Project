"""Run the official TAG-CF method on our datasets, comparable to the E0 table.

Protocol (matches the paper):
  1. Train a plain MF with BPR (no graph during training).
  2. Evaluate MF as-is (baseline).
  3. Grid-search TAG-CF's (m, n) on the validation set; apply the single
     test-time message-passing aggregation; evaluate on test with the best (m, n).

Uses our data split + evaluate_official (train+valid masking) so numbers line up
with the E0 comparison.

Usage:
  python -m experiments.exp_tagcf --dataset amazon_beauty,movielens_1m --seeds 42 --epochs 150 --patience 15
"""

import argparse

import pandas as pd

from experiments.common import add_common_args, comma_list, ensure_dirs, git_commit, int_list
from gnn_recommendation.config import DEVICE, make_config, set_global_seed
from gnn_recommendation.official import GraphRecDataset, load_official_modules
from gnn_recommendation.pipeline import prepare_dataset
from gnn_recommendation.tagcf import TAGCF_SEARCH_SPACE, MFTagCF
from gnn_recommendation.training import METRIC_COLUMNS, evaluate_official, train_bpr_model


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--n-layers", type=int, default=1, help="TAG-CF message-passing layers")
    parser.add_argument("--output", default="results/tagcf_comparison.csv")
    args = parser.parse_args()

    datasets = comma_list(args.dataset)
    seeds = [42] if args.quick else int_list(args.seeds)
    epochs = 2 if args.quick else args.epochs
    search_space = [-0.5, 0.0] if args.quick else TAGCF_SEARCH_SPACE

    ensure_dirs()
    commit = git_commit()
    rows = []

    for dataset_name in datasets:
        for seed in seeds:
            set_global_seed(seed)
            config = make_config(embed_dim=64)
            world, BasicDataset, PureMF, LightGCN, utils, Procedure = load_official_modules(
                "LightGCN-PyTorch", config
            )
            world.config = config
            world.seed = seed
            utils.set_seed(seed)

            _, train_df, train_pairs, valid_pairs, test_pairs, graph, n_users, n_items = prepare_dataset(
                dataset_name
            )
            dataset = GraphRecDataset.create(
                BasicDataset, n_users, n_items, train_pairs, test_pairs, graph, valid_pairs=valid_pairs
            )

            # 1. Train plain MF (no graph) with BPR.
            model = MFTagCF(config, dataset, n_layers=args.n_layers).to(DEVICE)
            model.use_mp = False
            bpr_loss = utils.BPRLoss(model, config)

            def val_cb(cur):
                if not dataset.validDict:
                    return 0.0
                cur.use_mp = False
                return evaluate_official(dataset, cur, utils, eval_dict=dataset.validDict, include_timing=False)["NDCG@10"]

            info = train_bpr_model(
                dataset, model, bpr_loss, Procedure, epochs,
                name=f"{dataset_name}/MF",
                log_every=max(epochs // 2, 1),
                eval_callback=val_cb if args.patience is not None else None,
                patience=args.patience,
            )

            # 2. MF baseline (no test-time MP).
            model.use_mp = False
            mf_test = evaluate_official(dataset, model, utils, eval_dict=dataset.testDict, exclude_dicts=[dataset.validDict])

            # 3. TAG-CF: grid-search (m, n) on validation, then test.
            model.use_mp = True
            best_val, best_mn = -1.0, (-0.5, -0.5)
            for mm in search_space:
                for nn_ in search_space:
                    model.m, model.n = mm, nn_
                    v = evaluate_official(dataset, model, utils, eval_dict=dataset.validDict, include_timing=False)["NDCG@10"]
                    if v > best_val:
                        best_val, best_mn = v, (mm, nn_)
            model.m, model.n = best_mn
            tag_test = evaluate_official(dataset, model, utils, eval_dict=dataset.testDict, exclude_dicts=[dataset.validDict])

            print(f"\n[{dataset_name}] seed={seed} | MF NDCG@10={mf_test['NDCG@10']:.4f} "
                  f"| TAG-CF NDCG@10={tag_test['NDCG@10']:.4f} (best m,n={best_mn}, val={best_val:.4f})")

            for tag, metrics, extra in [
                ("MF", mf_test, {}),
                ("MF+TAG-CF", tag_test, {"tagcf_m": best_mn[0], "tagcf_n": best_mn[1], "n_layers": args.n_layers}),
            ]:
                row = {"dataset": dataset_name, "model": tag, "seed": seed,
                       "train_seconds": info["train_seconds"], "git_commit": commit}
                row.update({k: metrics[k] for k in METRIC_COLUMNS if k in metrics})
                row.update({k: metrics[k] for k in ("infer_ms_per_user",) if k in metrics})
                row.update(extra)
                rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"\nWrote TAG-CF results to {args.output}")
    show = [c for c in ["dataset", "model", "Recall@10", "NDCG@10", "tagcf_m", "tagcf_n"] if c in out.columns]
    print(out[show].to_string(index=False))


if __name__ == "__main__":
    main()
