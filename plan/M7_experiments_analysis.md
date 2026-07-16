# M7 — Experiments & Analysis

Tạo thư mục `experiments/`, mỗi script ghi CSV vào `results/`, hình vào `results/figures/`.

```
experiments/
  common.py              # load data 1 lần, cache preprocessing+graph, build model theo tên/config
  exp_main.py            # E0 bảng so sánh chính
  exp_layers.py          # E1 sweep số sheaf layer
  exp_dim.py             # E2 sweep stalk/latent dim
  exp_expressiveness.py  # E3 gcn/gat/full sheaf
  exp_edges.py           # E4 binary/rating/time edges
  run_all.sh             # gọi E0–E4, có --quick
```

`common.py` cache phần preprocessing + graph (nặng), chỉ thay model/config giữa run.

## E0 — Bảng so sánh chính

- Dataset: Amazon All Beauty (chính) + MovieLens 1M (phụ).
- Models: Popularity, LightGCN, NGCF (+GAT), Sheaf4Rec(full_sheaf) [+PureMF phụ].
- Metric: full `METRIC_COLUMNS` + train/infer time, mean±std trên 3 seed.
- Output: `results/main_comparison.csv`. Mở rộng trực tiếp `pipeline.run_multi_dataset`.

## E1 — Sweep số sheaf layer (chống over-smoothing)

- Sheaf4Rec(full_sheaf), `stalk_dim` cố định; `n_layers ∈ {1,2,3,4,5,6,8}`.
- Chạy kèm LightGCN cùng dải layer để đối chiếu (kỳ vọng: sheaf không sụp khi sâu, LightGCN over-smooth).
- Output `results/layer_sweep.csv`; hình NDCG@10 vs #layers (Sheaf vs LightGCN).

## E2 — Sweep stalk/latent dim (accuracy–cost)

- Sheaf `stalk_dim ∈ {1,2,3,4,6,8}`; LightGCN/NGCF `latent_dim ∈ {16,32,64,128}` để có trục cost tương đương.
- Trục x = #params hoặc `infer_ms_per_user`; trục y = NDCG@10.
- Output `results/dim_sweep.csv`; hình accuracy–cost curve.

## E3 — Expressiveness study

- Cùng data/split/dim, đổi `restriction_type`: `gcn_like (N,1)` vs `gat_like (1,N)` vs `full_sheaf (N,N)`.
- Mục tiêu: full sheaf mới tạo cải thiện, không chỉ do capacity → cân #params càng gần càng tốt, ghi rõ.
- Output `results/expressiveness.csv`; hình bar NDCG@10/Recall@10.

## E4 — Rating/time-aware edges

- Model: Sheaf4Rec(full_sheaf) + LightGCN. `edge_mode ∈ {binary, rating, time}` (M4).
- Với `rating`: đặt `positive_threshold=None` (M3) để rating mang thông tin; ghi rõ setup.
- Output `results/edge_construction.csv`; hình bar 3 edge_mode.

## Tái lập

- Mỗi `exp_*.py` có `argparse` (dataset, seeds, epochs, output), mặc định Amazon.
- CSV kèm cột config (dataset, model, seed, n_layers, stalk_dim, restriction_type, edge_mode, epochs)
  + `git rev-parse HEAD`.
- `run_all.sh --quick` chạy ít epoch để smoke toàn bộ.

## Checklist M7

- [x] `common.py` (cache data/graph, build model theo tên/config).
- [x] E0–E4.
- [x] `run_all.sh` + `--quick`.
- [x] `train.sh`: full training script (GPU, 3 seeds, patience=10, epochs 100/50, log per experiment).
- [x] **Smoke**: `run_all.sh --quick` pass sạch E0–E4 trên GPU (Amazon Beauty, 1 epoch, 1 seed).
