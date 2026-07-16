# M6 — Evaluation protocol & Metrics

Sửa `gnn_recommendation/training.py`. Có thể làm **song song với M5** và phải xong trước M7.

## 6.1 Bổ sung F1@K và MRR@K

`evaluate_official` đã có recall/precision/ndcg/HitRatio@10. Thêm:

**F1@K** (từ P, R sẵn có):
```python
p, r = metrics[f"Precision@{k}"], metrics[f"Recall@{k}"]
metrics[f"F1@{k}"] = 2*p*r/(p+r) if (p+r) > 0 else 0.0
```

**MRR@K** (dùng ma trận hit `r` từ `utils.getLabel`):
```python
def mrr_at_k(r, k):
    r_k = r[:, :k]
    rr = (r_k / np.arange(1, k+1)).max(axis=1)   # reciprocal rank hit đầu tiên
    return rr.sum()
metrics[f"MRR@{k}"] = mrr_at_k(r, k) / len(users)
```

**HitRatio@K tổng quát**: mở rộng vòng `for k in ks` (không chỉ @10).

`METRIC_COLUMNS`:
```python
["Recall@10","Recall@20","Precision@10","Precision@20",
 "NDCG@10","NDCG@20","F1@10","F1@20","MRR@10","MRR@20",
 "HitRatio@10","HitRatio@20"]
```
Số chính báo cáo (in đậm): **Recall@10, NDCG@10**.

## 6.2 Đo thời gian (proposal: "slower to train, faster to serve")

- **Train time**: bọc `train_bpr_model` bằng `time.perf_counter()` → `train_seconds`.
- **Inference time** trong `evaluate_official`, phần forward + `getUsersRating` + `topk`:
  - `infer_total_s`, `infer_ms_per_user = infer_total_s/n_users*1000`.
  - Sheaf4Rec: đo riêng 1 lần `forward` (diffusion) — chi phí chính.
- Nối vào `results_df` trong `pipeline.py`. Ghi rõ device (`config.DEVICE`); nếu CPU-only, note.

## 6.3 Eval trên validation/test + model selection

- `evaluate_official(dataset, model, utils, eval_dict=None)`; mặc định `testDict`, truyền
  `validDict` để chấm val.
- Loại item train (`allPos`) khi rank; khi chấm test, loại cả train+val (nhất quán mọi model,
  ghi rõ quy ước trong báo cáo).
- **Model selection**: chọn hyperparameter/epoch theo **NDCG@10 trên val**, chấm test 1 lần.
- **Early stopping**: dừng nếu val NDCG@10 không cải thiện sau `patience` lần eval.

## 6.4 Bất định

- Mỗi cấu hình chạy ≥3 seed (`{42,43,44}`), báo cáo mean±std cho Recall@10, NDCG@10; lưu raw
  per-seed để vẽ error bar. Thêm `--seeds` vào runner.

## Checklist M6

- [x] F1@K, MRR@K, HitRatio@K tổng quát; cập nhật `METRIC_COLUMNS`.
- [x] Train time + inference time (tổng + per-user) → pipeline.
- [x] `eval_dict` cho val/test; loại train(+val) khi rank; early stopping + model selection.
- [x] Multi-seed mean/std.
- [x] **Verify**: F1 = 2PR/(P+R) trên vài ví dụ tay.
