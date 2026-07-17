# M7 — Experiments & Analysis

Mỗi script trong `experiments/` ghi CSV vào `results/`, hình vào `results/figures/`
(qua `experiments/build_report.py`, xem M8).

```
experiments/
  common.py               # add_common_args, run_configs, summarize_mean_std, FIVE_MODELS
  exp_main.py              # E0 bảng so sánh chính
  exp_layers.py            # E1 sweep n_layers
  exp_dim.py               # E2 sweep latent_dim (+ sheaf_stalk_dim riêng)
  exp_expressiveness.py    # E3 gcn_like/gat_like/full_sheaf (chỉ Sheaf4Rec)
  exp_edges.py              # E4 binary/rating/time edges
  exp_sota.py               # 5 model self-supervised phụ (SGL/SimGCL/DirectAU/NCL/LightGCL)
  exp_tagcf.py               # MF + TAG-CF (test-time message passing)
  exp_puremf.py              # PureMF baseline
  exp_tune_official.py       # grid-search hyperparameter cho Sheaf4Rec-official
  build_report.py            # gộp mọi CSV thành bảng + biểu đồ cuối (M8)
```

`experiments/common.py:run_configs` cache preprocessing + graph trong process (qua
`pipeline._PREPARED_DATASET_CACHE`, key gồm `dataset_name, min_interactions,
positive_threshold, edge_mode`), chỉ train lại model khi cấu hình data không đổi.

## Model mặc định (`--models`)

Mọi script E0/E1/E2/E4 nhận cờ `--models` (mặc định `FIVE_MODELS` =
`LightGCN,NGCF,Sheaf4Rec-official,NCL,DirectAU`, xem `experiments/common.py`), cho phép
thu hẹp/mở rộng danh sách model khi chạy 1 kịch bản:

```bash
python -m experiments.exp_main --models LightGCN,NGCF   # chỉ 2 model
```

**E3 là ngoại lệ**: không có `--models`, vì restriction-map expressiveness chỉ có ý nghĩa
với Sheaf4Rec (LightGCN/NGCF/NCL/DirectAU không có khái niệm restriction map).

## E0 — Bảng so sánh chính

- Dataset: Amazon Beauty + MovieLens 1M (mặc định `DATASET_MAIN` trong `train.sh`).
- Models: `FIVE_MODELS` (LightGCN, NGCF, Sheaf4Rec-official, NCL, DirectAU).
- Metric: full `METRIC_COLUMNS` + train/infer time, mean±std trên 3 seed.
- Output: `results/main_comparison.csv` (+ `_raw.csv` per-seed).

## E1 — Sweep số lớp truyền tin (`n_layers`)

- `n_layers ∈ {1,2,3,4,5,6,8}`, áp dụng cho mọi model trong `--models` (không chỉ
  Sheaf4Rec — `n_layers` map vào `lightGCN_n_layers`/`sheaf_n_layers` tùy model).
- Output `results/layer_sweep.csv`; vẽ NDCG@10 vs #layers, so sánh giữa các model.

## E2 — Sweep embedding dimension

- `latent_dim ∈ {16,32,64,128}` — áp dụng cho mọi model trong `--models`.
- Riêng khi `Sheaf4Rec-official` nằm trong `--models`: thêm sweep phụ
  `sheaf_stalk_dim ∈ {1,2,3,4,6,8}` (kiến trúc-đặc-thù, tách khỏi `latent_dim`).
- Output `results/dim_sweep.csv`.

## E3 — Sheaf4Rec expressiveness study (chỉ Sheaf4Rec, dùng bản `sheaf.py`)

- Cố định data/split/dim, đổi `restriction_type`: `Sheaf4Rec-gcn_like` (0 params) vs
  `Sheaf4Rec-gat_like` (81 params) vs `Sheaf4Rec-full_sheaf` (216 params).
- Mục tiêu: chứng minh cải thiện đến từ cấu trúc sheaf, không chỉ từ số tham số nhiều hơn.
- Output `results/expressiveness.csv`.

## E4 — Rating/time-aware edges

- `edge_mode ∈ {binary, rating, time}`, áp dụng cho mọi model trong `--models`.
- Với `rating`: nên đặt `positive_threshold=None` (M3) để rating còn mang thông tin phân
  biệt (nếu không, mọi cạnh dương chỉ còn giá trị 4 hoặc 5).
- Output `results/edge_construction.csv`.

## Model phụ — 3 script riêng (không thuộc E0-E4)

- **`exp_sota.py`**: train 5 model self-supervised (SGL, SimGCL, DirectAU, NCL, LightGCL)
  — lưu ý DirectAU/NCL cũng nằm trong `FIVE_MODELS` của E0, script này train lại chúng
  cùng nhóm 3 model phụ (SGL/SimGCL/LightGCL) để so sánh nội bộ nhóm self-supervised.
  Output `results/sota_comparison.csv`.
- **`exp_tagcf.py`**: train MF (BPR thuần, không graph) rồi áp 1 bước test-time
  message-passing (grid-search hệ số chuẩn hóa `(m,n)` trên validation). Output
  `results/tagcf_comparison.csv` (2 dòng mỗi seed: `MF` và `MF+TAG-CF`).
- **`exp_puremf.py`**: train PureMF (import trực tiếp từ LightGCN-PyTorch gốc). Output
  `results/puremf_comparison.csv`.
- **`exp_tune_official.py`**: grid-search `(n_layers, latent_dim, lr)` cho
  `Sheaf4Rec-official`, chọn cấu hình tốt nhất theo validation NDCG@10. Output vào
  `results/tune/` (giữ nguyên qua `.gitignore`, không bị dọn tự động).

## Tái lập

- Mọi `exp_*.py` dùng `argparse` qua `add_common_args` (`--dataset`, `--seeds`, `--epochs`,
  `--patience`, `--models` nếu áp dụng), mặc định Amazon Beauty.
- CSV kèm cột config (`dataset`, `model`, `seed`, các cột config đặc thù như `n_layers`,
  `latent_dim`, `sheaf_stalk_dim`, `edge_mode`, `epochs`) + `git_commit`.
- `--quick`: override 1-2 epoch, 1 seed — dùng để smoke-test toàn bộ script trước khi
  chạy full (xem `test.sh --smoke` ở M8).

## Checklist M7

- [x] `common.py`: cache data/graph, `FIVE_MODELS`, `run_configs`/`summarize_mean_std`.
- [x] E0–E4, mỗi script hỗ trợ `--models` (trừ E3).
- [x] `exp_sota.py`, `exp_tagcf.py`, `exp_puremf.py`, `exp_tune_official.py`.
- [x] `train.sh`: entry point tổng — mỗi giai đoạn (`e0`..`e4`, `sota`, `tagcf`, `puremf`,
      `all`) chạy được độc lập qua tag, override `MODELS=`/`DATASET_MAIN=`/`SEEDS=` qua
      biến môi trường.
- [x] **Smoke**: `bash test.sh --smoke` chạy `--quick` qua toàn bộ 8 script rồi build report
      — dùng để kiểm tra pipeline không lỗi trước khi train full.
