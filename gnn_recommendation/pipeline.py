"""End-to-end experiment pipeline."""

import pandas as pd

from .config import DEVICE, EMBED_DIM, MIN_INTERACTIONS, N_EPOCHS_MULTI, N_LAYERS, SEED, make_config, set_global_seed
from .data import DATASET_REGISTRY, load_dataset_by_name
from .extra_models import build_extra_models
from .graph import build_normalized_graph, compute_edge_weight
from .official import GraphRecDataset, load_official_modules
from .preprocessing import (
    global_timestamp_split,
    pairs_from_frame,
    preprocess,
    quantile_timestamp_cutoffs,
)
from .training import METRIC_COLUMNS, evaluate_official, train_bpr_model

_PREPARED_DATASET_CACHE = {}

# Amazon Reviews 2023's own official "timestamp" benchmark split cutoffs (ms epoch),
# reverse-engineered from data/amazon_reviews_2023/benchmark/0core/timestamp/*.csv
# (train max ts=1628643321568, valid range=[1628643415648, 1658002676817], test
# min ts=1658002770869) -- matches the paper's documented t1/t2 exactly.
#
# Unit warning: these are hardcoded ms-epoch, valid ONLY for Amazon datasets.
# Any other dataset goes through quantile_timestamp_cutoffs() below instead,
# which derives t1/t2 from that dataset's own timestamp column (whatever unit
# it uses -- e.g. MovieLens ratings.dat is s-epoch). Nothing here asserts unit
# consistency; adding a new Amazon-like dataset with ms timestamps must NOT
# reuse these two constants directly.
AMAZON_GLOBAL_TS_T1 = 1628643414042
AMAZON_GLOBAL_TS_T2 = 1658002729837


def prepare_dataset(
    dataset_name,
    min_interactions=MIN_INTERACTIONS,
    positive_threshold: float | None = 4.0,
    edge_mode: str = "binary",
):
    cache_key = (dataset_name, min_interactions, positive_threshold, edge_mode)
    if cache_key in _PREPARED_DATASET_CACHE:
        print(f"[{dataset_name}] Reusing cached preprocessed data/graph for {cache_key}")
        return _PREPARED_DATASET_CACHE[cache_key]

    data = load_dataset_by_name(dataset_name)
    data_filtered, _, _, n_users, n_items = preprocess(
        data,
        min_interactions=min_interactions,
        positive_threshold=positive_threshold,
    )
    print(f"[{dataset_name}] Filtered: {n_users} users, {n_items} items, {len(data_filtered)} interactions")

    if dataset_name == "amazon_beauty":
        t1, t2 = AMAZON_GLOBAL_TS_T1, AMAZON_GLOBAL_TS_T2
    else:
        t1, t2 = quantile_timestamp_cutoffs(data_filtered)
    print(f"[{dataset_name}] Global timestamp split: t1={t1:.0f}, t2={t2:.0f}")
    train_df, valid_pairs, test_pairs = global_timestamp_split(data_filtered, t1, t2)
    train_df_for_graph = train_df.drop_duplicates(subset=["u_idx", "i_idx"], keep="last").reset_index(drop=True)
    train_pairs = pairs_from_frame(train_df_for_graph)
    print(
        f"[{dataset_name}] Split: {len(train_pairs)} train pairs, "
        f"{len(valid_pairs)} valid pairs, {len(test_pairs)} test pairs"
    )
    edge_weight = compute_edge_weight(train_df_for_graph, edge_mode=edge_mode)
    graph = build_normalized_graph(
        n_users,
        n_items,
        train_pairs,
        DEVICE,
        edge_weight=edge_weight,
        edge_mode=edge_mode,
    )
    prepared = (
        data_filtered,
        train_df_for_graph,
        train_pairs,
        valid_pairs,
        test_pairs,
        graph,
        n_users,
        n_items,
    )
    _PREPARED_DATASET_CACHE[cache_key] = prepared
    return prepared


def build_all_models(config, dataset, PureMF, LightGCN, include_gat=True, include_auxiliary=True):
    models = {
        "LightGCN": LightGCN(config, dataset).to(DEVICE),
        **build_extra_models(
            config,
            dataset,
            DEVICE,
            include_gat=include_gat,
            include_auxiliary=include_auxiliary,
        ),
    }
    if include_auxiliary:
        models = {"PureMF": PureMF(config, dataset).to(DEVICE), **models}
    return models


def _filter_models(models: dict, model_names: list[str] | None = None) -> dict:
    if not model_names:
        return models
    missing = [name for name in model_names if name not in models]
    if missing:
        raise ValueError(f"Requested model(s) not available: {missing}. Available: {sorted(models)}")
    return {name: models[name] for name in model_names}


def run_full_pipeline(
    dataset_name: str,
    min_interactions: int = MIN_INTERACTIONS,
    embed_dim: int = EMBED_DIM,
    n_layers: int = N_LAYERS,
    epochs: int = N_EPOCHS_MULTI,
    include_gat: bool = True,
    repo_dir: str = "LightGCN-PyTorch",
    positive_threshold: float | None = 4.0,
    edge_mode: str = "binary",
    include_auxiliary: bool = True,
    seed: int = SEED,
    patience: int | None = None,
    model_names: list[str] | None = None,
    config_overrides: dict | None = None,
) -> pd.DataFrame:
    config = make_config(embed_dim=embed_dim, n_layers=n_layers)
    if config_overrides:
        config.update(config_overrides)
    world, BasicDataset, PureMF, LightGCN, utils, Procedure = load_official_modules(repo_dir, config)
    world.dataset = dataset_name
    world.config = config
    world.seed = seed
    set_global_seed(seed)

    _, train_df, train_pairs, valid_pairs, test_pairs, graph, n_users, n_items = prepare_dataset(
        dataset_name,
        min_interactions=min_interactions,
        positive_threshold=positive_threshold,
        edge_mode=edge_mode,
    )
    dataset = GraphRecDataset.create(
        BasicDataset,
        n_users,
        n_items,
        train_pairs,
        test_pairs,
        graph,
        valid_pairs=valid_pairs,
    )
    dataset.train_df = train_df
    utils.set_seed(seed)
    models = build_all_models(
        config,
        dataset,
        PureMF,
        LightGCN,
        include_gat=include_gat,
        include_auxiliary=include_auxiliary,
    )
    models = _filter_models(models, model_names)

    results = {}
    for name, model in models.items():
        train_seconds = 0.0
        best_val_ndcg10 = None
        if getattr(model, "trainable", True):
            bpr_loss = utils.BPRLoss(model, config)
            def val_score_callback(current_model):
                if not dataset.validDict:
                    return 0.0
                val_metrics = evaluate_official(
                    dataset,
                    current_model,
                    utils,
                    eval_dict=dataset.validDict,
                    include_timing=False,
                )
                return val_metrics["NDCG@10"]

            train_info = train_bpr_model(
                dataset,
                model,
                bpr_loss,
                Procedure,
                epochs,
                name=f"{dataset_name}/{name}",
                log_every=max(epochs // 2, 1),
                eval_callback=val_score_callback if patience is not None else None,
                patience=patience,
            )
            train_seconds = train_info["train_seconds"]
            best_val_ndcg10 = train_info["best_val_ndcg10"]
            if best_val_ndcg10 == -float("inf"):
                best_val_ndcg10 = None
        else:
            print(f"[{dataset_name}/{name}] Non-trainable baseline; skipping BPR training.")
        val_metrics = (
            evaluate_official(dataset, model, utils, eval_dict=dataset.validDict)
            if dataset.validDict
            else {}
        )
        # Test ranking excludes train positives through allPos and validation positives explicitly.
        test_metrics = evaluate_official(
            dataset,
            model,
            utils,
            eval_dict=dataset.testDict,
            exclude_dicts=[dataset.validDict],
        )
        row = dict(test_metrics)
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["train_seconds"] = train_seconds
        row["best_val_NDCG@10"] = best_val_ndcg10 if best_val_ndcg10 is not None else row.get("val_NDCG@10", 0.0)
        row["device"] = str(DEVICE)
        results[name] = row

    results_df = pd.DataFrame(results).T
    results_df.index.name = "model"
    ordered = [col for col in METRIC_COLUMNS if col in results_df.columns]
    timing_cols = ["infer_total_s", "infer_ms_per_user", "train_seconds", "best_val_NDCG@10", "device"]
    val_cols = [f"val_{col}" for col in METRIC_COLUMNS if f"val_{col}" in results_df.columns]
    results_df = results_df[ordered + val_cols + [col for col in timing_cols if col in results_df.columns]]
    for col in results_df.columns:
        if col != "device":
            results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
    results_df["seed"] = seed
    return results_df


def run_multi_dataset(
    dataset_names=None,
    output_csv: str = "multi_dataset_results.csv",
    seeds=None,
    **pipeline_kwargs,
) -> pd.DataFrame:
    dataset_names = dataset_names or list(DATASET_REGISTRY)
    seeds = seeds or [SEED]
    all_results = {}
    for seed in seeds:
        for dataset_name in dataset_names:
            display_name = DATASET_REGISTRY[dataset_name]["display_name"]
            print(f"\n{'=' * 60}\nRUNNING PIPELINE FOR: {display_name} | seed={seed}\n{'=' * 60}")
            all_results[(dataset_name, seed)] = run_full_pipeline(dataset_name, seed=seed, **pipeline_kwargs)

    raw = pd.concat(all_results, names=["dataset", "seed", "model"])
    if len(seeds) == 1:
        single = raw.droplevel("seed")
        single.to_csv(output_csv)
        print(f"\nSaved combined results to {output_csv}")
        return single

    raw_path = output_csv.replace(".csv", "_raw.csv")
    raw.to_csv(raw_path)
    numeric = raw.select_dtypes(include="number").drop(
        columns=[col for col in ("seed",) if col in raw.columns],
        errors="ignore",
    )
    grouped = numeric.groupby(level=["dataset", "model"])
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std(ddof=0).add_suffix("_std")
    combined = pd.concat([mean_df, std_df], axis=1)
    combined.to_csv(output_csv)
    print(f"\nSaved multi-seed raw results to {raw_path}")
    print(f"Saved multi-seed mean/std results to {output_csv}")
    return combined
