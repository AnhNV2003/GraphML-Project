# Plan hoàn thiện project — Sheaf4Rec trên Amazon Reviews 2023 (All Beauty) + MovieLens 1M

Kế hoạch code đưa repo từ scaffold hiện tại lên đúng phạm vi proposal:

> **Sheaf Neural Networks for E-Commerce Product Recommendation: Applying Sheaf4Rec
> to Amazon Reviews 2023** (`Formatting_Instructions_For_NeurIPS_2025.pdf`).

Cấu trúc theo **phase ML tuần tự M1–M8** (tailored cho graph ML / recommender research, không
phải tabular CRISP-DM: "Feature Engineering" được thay bằng "Graph Construction"; phần model +
evaluation + experiments được tách riêng vì đó là đóng góp khoa học chính).

## Phase

| Phase | File | Nội dung |
|---|---|---|
| M1 | [M1_problem_and_data.md](M1_problem_and_data.md) | Problem framing + data acquisition (đọc local, registry) |
| M2 | [M2_eda.md](M2_eda.md) | EDA trên real data, số liệu cho báo cáo, biện minh ngưỡng rating |
| M3 | [M3_preprocessing.md](M3_preprocessing.md) | Dedup, k-core, ngưỡng `rating≥4`, split train/val/test |
| M4 | [M4_graph_construction.md](M4_graph_construction.md) | Bipartite graph: binary / rating-aware / time-aware edges |
| M5 | [M5_models_sheaf4rec.md](M5_models_sheaf4rec.md) | Popularity, LightGCN/NGCF, **Sheaf4Rec thật** + expressiveness |
| M6 | [M6_evaluation_metrics.md](M6_evaluation_metrics.md) | F1@K, MRR@K, NDCG, timing, eval val/test, multi-seed |
| M7 | [M7_experiments_analysis.md](M7_experiments_analysis.md) | E0 bảng chính + E1–E4 (layer/dim/expressiveness/edge) |
| M8 | [M8_results_report.md](M8_results_report.md) | Bảng + hình NeurIPS, cập nhật docs |

## Dataset (đã tải sẵn local — không cần tải mạng)

- **Amazon Reviews 2023 – All Beauty** (chính):
  - Raw: `data/amazon_reviews_2023/raw/review_categories/All_Beauty.jsonl` (~700K review).
  - Meta: `data/amazon_reviews_2023/raw/meta_categories/meta_All_Beauty.jsonl`.
  - Split chuẩn của tác giả: `data/amazon_reviews_2023/benchmark/{0core,5core}/{last_out,timestamp}/All_Beauty.{train,valid,test}.csv`
    (cột `user_id, parent_asin, rating, timestamp`; **timestamp mili-giây**).
- **MovieLens 1M** (phụ): `data/datamovielens-1m/ratings.dat` (`UserID::MovieID::Rating::Timestamp`) + `movies.dat`, `users.dat`.

## Definition of done

1. Chạy trên **real data local** (Amazon chính, MovieLens phụ), số báo cáo có `is_real_data=True`.
2. So sánh 3 tầng model: **Popularity** / **LightGCN**+**NGCF** / **Sheaf4Rec** (sheaf thật).
3. Đủ metric: Recall/Precision/NDCG/HitRatio/**F1**/**MRR** @K, + **train & inference time**;
   số chính là Recall@10, NDCG@10.
4. Đủ 4 phân tích: sheaf-layer sweep, stalk/latent-dim sweep, expressiveness `(N,1)/(1,N)/(N,N)`,
   rating/time-aware edges.
5. Sinh bảng + hình cho báo cáo NeurIPS.

## Thứ tự phụ thuộc

```
M1 → M2 → M3 → M4 → M5 ┐
                        ├→ M7 → M8
        M6 (song song với M5, xong trước M7)
```
Không sinh số đáng tin trước khi có real data + split (M1–M3) và metric đầy đủ (M6).
Sheaf4Rec (M5) rủi ro cao nhất → nên có M6 làm khung kiểm thử sẵn.

## Gap chính so với code hiện tại

| Gap | Hiện trạng | Phase |
|---|---|---|
| Sheaf4Rec thật (stalk `d`, restriction matrix, sheaf Laplacian) | Chỉ stub scalar `d=1` ([extra_models.py:84](../gnn_recommendation/extra_models.py#L84)) | M5 |
| Expressiveness `(N,1)/(1,N)/(N,N)` | Không có | M5 |
| Popularity baseline | Chỉ có PureMF | M5 |
| Ngưỡng implicit `rating≥4` | Giữ mọi rating | M3 |
| Validation split | Chỉ train/test | M3 |
| F1@K, MRR@K, timing | Thiếu | M6 |
| Chạy real data (đang synthetic) + đọc local | `is_real_data=False`, loader qua URL | M1 |
| Rating/time-aware edges | Chỉ nhị phân | M4 |
| Harness 4 phân tích | Không có | M7 |

## Nguyên tắc chung

- **Không phá pipeline hiện tại** — thêm qua flag/option; `run_pipeline.py` vẫn chạy như cũ.
- **Mọi model theo API `BPRModelBase`** (`forward -> (user_emb,item_emb)`, `bpr_loss`,
  `getUsersRating`) để tái dùng `Procedure`/`utils` của LightGCN-PyTorch.
- Reproducibility: set seed qua `set_global_seed`, ghi config + `git HEAD` cạnh mỗi kết quả.
- Tách số thật vs smoke test qua cột `is_real_data`.

## Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Sheaf Laplacian sparse tốn RAM `d²·|E|` trên MovieLens | Bắt đầu `d≤4`, k-core mạnh hơn; dim lớn chỉ chạy Amazon |
| All Beauty 5-core nhỏ (~2k train) → số nhiễu | Dùng thêm 0core / raw + k-core vừa phải; báo cáo cả hai |
| Sheaf4Rec khó khớp paper 100% | Ghi rõ "faithful re-implementation, sai khác X"; đối chiếu NSD official code |
| torch-geometric khó cài (GAT) | GAT optional (`--no-gat`); NGCF là "standard GNN" thay thế |
| timestamp Amazon là ms | Xử lý nhất quán, không trộn giây/ms |
