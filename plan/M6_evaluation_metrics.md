# M6 — Evaluation protocol & Metrics

Code trong `gnn_recommendation/training.py`.

## 6.1 Metric đầy đủ (`evaluate_official`)

`METRIC_COLUMNS`:
```python
["Recall@10","Recall@20","Precision@10","Precision@20",
 "NDCG@10","NDCG@20","F1@10","F1@20","MRR@10","MRR@20",
 "HitRatio@10","HitRatio@20"]
```
Số chính báo cáo (in đậm): **Recall@10, NDCG@10**.

- **F1@K**: `2·P·R/(P+R)` từ Precision/Recall đã tính sẵn.
- **MRR@K**: `mrr_at_k(r, k)` — reciprocal rank của hit đầu tiên trong top-K, trung bình
  trên toàn bộ user.
- **HitRatio@K**: tỷ lệ user có ít nhất 1 hit trong top-K.
- Full-catalog ranking: loại `allPos` (train) + `exclude_dicts` (thường là `validDict`
  khi chấm test) khỏi rating trước khi lấy top-K — không dùng sampled-negative ranking.

## 6.2 Đo thời gian

- **Train time**: `train_bpr_model` bọc bằng `time.perf_counter()` → `train_seconds`.
- **Inference time**: trong `evaluate_official`, đo riêng phần `getUsersRating` +
  `torch.topk` mỗi batch → `infer_total_s`, `infer_ms_per_user = infer_total_s/n_users*1000`.
- Ghi kèm `device` (`gnn_recommendation.config.DEVICE`) vào mỗi dòng kết quả.

## 6.3 Eval trên validation/test + model selection

- `evaluate_official(dataset, model, utils, eval_dict=None)` — mặc định `testDict`,
  truyền `validDict` để chấm val.
- Test ranking loại cả `allPos` (train) **và** `validDict` (`exclude_dicts=[dataset.validDict]`)
  — nhất quán giữa mọi model, tránh rò rỉ thông tin từ validation set vào test ranking.
- **Model selection**: `train_bpr_model` theo dõi `val_NDCG@10` mỗi epoch qua
  `eval_callback`, giữ lại `best_state` (checkpoint có val NDCG@10 cao nhất).
- **Early stopping**: dừng nếu val NDCG@10 không cải thiện sau `patience` lần eval liên
  tiếp (mặc định `patience=10` trong `train.sh`).

## 6.4 Sửa lỗi parse loss (quan trọng cho DirectAU)

`train_bpr_model` parse chuỗi loss trả về từ `Procedure.BPR_train_original` bằng regex:

```python
_LOSS_RE = re.compile(r"loss(-?\d+\.\d+)")
loss_val = float(_LOSS_RE.search(out).group(1))
```

Cách cũ (`out.split("loss")[1].split("-")[0]`) crash với loss **âm** — cần thiết cho
**DirectAU**, vì alignment+uniformity loss của nó hợp lệ khi âm (uniformity term là log
của một giá trị `<1`). Không có fix này, DirectAU không train được.

## 6.5 Bất định (multi-seed)

Mỗi cấu hình chạy `SEEDS=42,43,44` (mặc định `train.sh`), báo cáo mean±std cho mọi
metric qua `experiments/common.py:summarize_mean_std`. File `_raw.csv` lưu kết quả từng
seed riêng lẻ để vẽ error bar; file chính (không hậu tố `_raw`) là bảng mean/std.

## Checklist M6

- [x] F1@K, MRR@K, HitRatio@K tổng quát; `METRIC_COLUMNS` đầy đủ.
- [x] Train time + inference time (tổng + per-user) → mỗi dòng kết quả.
- [x] `eval_dict` cho val/test; loại train(+val) khi rank; early stopping + model selection.
- [x] Multi-seed mean/std (`summarize_mean_std`).
- [x] Regex-based loss parsing (hỗ trợ loss âm — cần cho DirectAU).
