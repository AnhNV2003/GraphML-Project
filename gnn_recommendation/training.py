"""Training and evaluation helpers using official LightGCN metrics/loss loops."""

import copy
import re
import time

import numpy as np
import torch

from .config import DEVICE

# Procedure.BPR_train_original returns f"loss{aver_loss:.3f}-{time_info}"; aver_loss
# can be negative (e.g. DirectAU's alignment+uniformity loss), so a plain
# out.split("loss")[1].split("-")[0] breaks on the sign. Match the float directly.
_LOSS_RE = re.compile(r"loss(-?\d+\.\d+)")

METRIC_COLUMNS = [
    "Recall@10",
    "Recall@20",
    "Precision@10",
    "Precision@20",
    "NDCG@10",
    "NDCG@20",
    "F1@10",
    "F1@20",
    "MRR@10",
    "MRR@20",
    "HitRatio@10",
    "HitRatio@20",
]


def mrr_at_k(r: np.ndarray, k: int) -> float:
    r_k = r[:, :k]
    reciprocal_ranks = (r_k / np.arange(1, k + 1)).max(axis=1)
    return float(reciprocal_ranks.sum())


def _merge_exclusions(dataset, users, exclude_dicts=None):
    all_pos = dataset.getUserPosItems(users)
    merged = [set(items.tolist() if hasattr(items, "tolist") else list(items)) for items in all_pos]
    for exclude_dict in exclude_dicts or []:
        for idx, user in enumerate(users):
            if user in exclude_dict:
                merged[idx].update(exclude_dict[user])
    return merged


def evaluate_official(
    dataset,
    model,
    utils,
    ks=(10, 20),
    batch_size: int = 1000,
    eval_dict=None,
    exclude_dicts=None,
    include_timing: bool = True,
) -> dict[str, float]:
    model.eval()
    target_dict = eval_dict if eval_dict is not None else dataset.testDict
    users = list(target_dict.keys())
    max_k = max(ks)

    all_topk = []
    ground_truth = []
    infer_total_s = 0.0

    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            batch_users = users[start : start + batch_size]
            batch_users_t = torch.tensor(batch_users, dtype=torch.long, device=DEVICE)
            infer_start = time.perf_counter()
            ratings = model.getUsersRating(batch_users_t).clone()

            all_pos = _merge_exclusions(dataset, batch_users, exclude_dicts=exclude_dicts)
            exclude_index, exclude_items = [], []
            for i, items in enumerate(all_pos):
                exclude_index += [i] * len(items)
                exclude_items += list(items)
            if exclude_index:
                ratings[exclude_index, exclude_items] = -(1 << 10)

            _, topk_idx = torch.topk(ratings, k=max_k)
            infer_total_s += time.perf_counter() - infer_start
            all_topk.append(topk_idx.cpu().numpy())
            ground_truth.extend([target_dict[u] for u in batch_users])

    topk_idx = np.concatenate(all_topk, axis=0)
    r = utils.getLabel(ground_truth, topk_idx)

    metrics = {}
    for k in ks:
        ret = utils.RecallPrecision_ATk(ground_truth, r, k)
        metrics[f"Recall@{k}"] = ret["recall"] / len(users)
        metrics[f"Precision@{k}"] = ret["precision"] / len(users)
        metrics[f"NDCG@{k}"] = utils.NDCGatK_r(ground_truth, r, k) / len(users)
        precision = metrics[f"Precision@{k}"]
        recall = metrics[f"Recall@{k}"]
        metrics[f"F1@{k}"] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        metrics[f"MRR@{k}"] = mrr_at_k(r, k) / len(users)
        hits = (r[:, :k].sum(axis=1) > 0).sum()
        metrics[f"HitRatio@{k}"] = hits / len(users)

    if include_timing:
        metrics["infer_total_s"] = infer_total_s
        metrics["infer_ms_per_user"] = infer_total_s / len(users) * 1000 if users else 0.0
    return metrics


def train_bpr_model(
    dataset,
    model,
    bpr_loss,
    Procedure,
    epochs,
    name="model",
    log_every=5,
    eval_callback=None,
    patience: int | None = None,
):
    history = []
    train_start = time.perf_counter()
    best_score = -float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(epochs):
        out = Procedure.BPR_train_original(dataset, model, bpr_loss, epoch)
        loss_val = float(_LOSS_RE.search(out).group(1))
        history.append(loss_val)
        val_score = None
        if eval_callback is not None:
            val_score = eval_callback(model)
            if val_score > best_score:
                best_score = val_score
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
        if epoch % log_every == 0 or epoch == epochs - 1:
            suffix = f" | val_NDCG@10={val_score:.5f}" if val_score is not None else ""
            print(f"[{name}] Epoch {epoch:3d} | {out}{suffix}")
        if patience is not None and eval_callback is not None and bad_epochs >= patience:
            print(f"[{name}] Early stopping at epoch {epoch} (best val_NDCG@10={best_score:.5f})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - train_start
    return {"history": history, "train_seconds": train_seconds, "best_val_ndcg10": best_score}
