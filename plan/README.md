# Plan — Sheaf4Rec vs 4 model đồ thị chuẩn trên Amazon Reviews 2023 + MovieLens 1M

> **Sheaf Neural Networks for E-Commerce Product Recommendation: Applying Sheaf4Rec
> to Amazon Reviews 2023** (`assets/papers/Formatting_Instructions_For_NeurIPS_2025.pdf`).

Cấu trúc theo **phase tuần tự M1–M8** (tailored cho graph ML / recommender research,
không phải tabular CRISP-DM: "Feature Engineering" được thay bằng "Graph Construction";
model + evaluation + experiments tách riêng vì đó là đóng góp khoa học chính).

**Lưu ý khi đọc:** các file M1-M8 dưới đây mô tả trạng thái **hiện tại** của code, không
phải nhật ký lịch sử. Dự án đã trải qua vài lần đổi hướng lớn so với kế hoạch ban đầu
(xem mục "Các quyết định thay đổi lớn" bên dưới) — nội dung mỗi phase đã được viết lại
để khớp code thật, không cần đối chiếu ngược lịch sử git.

## Phase

| Phase | File | Nội dung |
|---|---|---|
| M1 | [M1_problem_and_data.md](M1_problem_and_data.md) | Problem framing + 2 dataset (Amazon Video Games, MovieLens 1M) |
| M2 | [M2_eda.md](M2_eda.md) | EDA trên real data, biện minh k-core/ngưỡng rating |
| M3 | [M3_preprocessing.md](M3_preprocessing.md) | Dedup, k-core (k=5), ngưỡng `rating≥4`, **global timestamp split** |
| M4 | [M4_graph_construction.md](M4_graph_construction.md) | Bipartite graph: binary / rating-aware / time-aware edges |
| M5 | [M5_models_sheaf4rec.md](M5_models_sheaf4rec.md) | 5 model chuẩn (LightGCN, NGCF, Sheaf4Rec-official, NCL, DirectAU) + model phụ |
| M6 | [M6_evaluation_metrics.md](M6_evaluation_metrics.md) | Recall/Precision/NDCG/F1/MRR/HitRatio@K, timing, val/test, multi-seed |
| M7 | [M7_experiments_analysis.md](M7_experiments_analysis.md) | E0 bảng chính + E1–E4 (layer/dim/expressiveness/edge) + SOTA/TAG-CF/PureMF |
| M8 | [M8_results_report.md](M8_results_report.md) | `train.sh`/`test.sh`, bảng + hình, Docker, báo cáo |

## 5 model chuẩn (trọng tâm so sánh)

| Model | Paper | Venue | Năm |
|---|---|---|---|
| LightGCN | He et al. | SIGIR | 2020 |
| NGCF | Wang et al. | SIGIR | 2019 |
| Sheaf4Rec | Purificato et al. | ACM TORS | 2023/2025 |
| NCL | Lin et al. | WWW | 2022 |
| DirectAU | Wang et al. | KDD | 2022 |

Mỗi model là **faithful port** từ repo gốc của tác giả (vendor trong `assets/external_repos/`),
không tự ý đơn giản hóa kiến trúc — xem M5 cho chi tiết verify từng model.

## Model phụ (không nằm trong bảng chính, chạy qua stage riêng)

- **Popularity** — baseline non-personalized (top-item phổ biến, không train).
- **PureMF** — matrix factorization thuần, import trực tiếp từ `LightGCN-PyTorch` gốc.
- **MF + TAG-CF** — MF + test-time message-passing (Test-time Aggregation for CF).
- **SGL, SimGCL, LightGCL** — 3 model self-supervised còn lại cùng họ với NCL/DirectAU,
  dùng chung encoder LightGCN-style, chỉ khác loss.
- **GAT** (optional, cần `torch-geometric`), **UltraGCN-stub** (MF trơn, KHÔNG phải
  UltraGCN thật — xem cảnh báo ở M5).

## Dataset

| Dataset | Vai trò | Nguồn local |
|---|---|---|
| Amazon Video Games | Chính | `assets/data/amazon_reviews_2023/raw/review_categories/Video_Games.jsonl` |
| MovieLens 1M | Phụ | `assets/data/datamovielens-1m/ratings.dat` |

Tải Amazon category bằng `python install_dataset_huggingface.py --category <Tên>`
(script giữ lại category đã tải trước đó, không ghi đè khi tải category mới).
MovieLens tự tải từ GroupLens nếu thiếu file local.

## Split protocol: chỉ còn 1 chiến lược — global timestamp split

Toàn bộ pipeline dùng **global timestamp split** (phong cách benchmark chính thức của
Amazon Reviews 2023): một cặp cutoff `(t1, t2)` áp dụng cho toàn dataset —
`train = t < t1`, `valid = t1 ≤ t < t2`, `test = t ≥ t2`.

> Chiến lược leave-one-out (per-user relative cut) đã được thử ở giai đoạn đầu dự án
> nhưng **đã bị loại bỏ hoàn toàn khỏi code** (không còn `leave_one_out_split` trong
> `preprocessing.py`) để tất cả kết quả trong `results/` nhất quán theo 1 giao thức duy nhất.

## Definition of done

1. Chạy trên **real data local** (không còn synthetic fallback — thiếu data thì báo lỗi
   thẳng thay vì âm thầm sinh số liệu giả).
2. Bảng so sánh chính: **5 model chuẩn** (LightGCN, NGCF, Sheaf4Rec-official, NCL, DirectAU)
   trên Amazon Video Games + MovieLens 1M.
3. Đủ metric: Recall/Precision/NDCG/HitRatio/F1/MRR @10/@20, + train & inference time;
   số chính là Recall@10, NDCG@10.
4. Đủ 4 phân tích ablation: layer sweep (E1), dim sweep (E2), Sheaf4Rec expressiveness (E3),
   edge-construction (E4) — tất cả hỗ trợ chọn model qua `--models`.
5. Sinh bảng + hình cho báo cáo qua `train.sh` + `test.sh`.
6. Đóng gói tái sử dụng qua Docker (`Dockerfile` + `docker-compose.yml`).

## Thứ tự phụ thuộc

```
M1 → M2 → M3 → M4 → M5 ┐
                        ├→ M7 → M8
        M6 (song song với M5, xong trước M7)
```

## Các quyết định thay đổi lớn so với kế hoạch ban đầu

| Quyết định | Trước | Sau | Lý do |
|---|---|---|---|
| Model trọng tâm | Popularity / LightGCN+NGCF / Sheaf4Rec (3 tầng) | 5 model đồ thị chuẩn có paper riêng (LightGCN, NGCF, Sheaf4Rec, NCL, DirectAU) | Thu hẹp về các model có tính học thuật/canonical rõ ràng, dễ trích dẫn |
| Split | Leave-one-out (per-user) | Global timestamp split (per-dataset) | Khớp benchmark chính thức Amazon Reviews 2023; LOO đã bị xóa khỏi code |
| k-core | `MIN_INTERACTIONS=2` | `MIN_INTERACTIONS=5` | Lọc mạnh hơn, giảm nhiễu từ user/item quá thưa |
| Sheaf4Rec | 1 bản tự viết (`sheaf.py`, 3 biến thể restriction) | Thêm bản port trung thành `sheaf_official.py` (`Sheaf4Rec-official`) dùng cho bảng chính; bản tự viết (`Sheaf4Rec-full_sheaf` etc.) chỉ dùng cho E3 | Đảm bảo số liệu bảng chính khớp kiến trúc gốc trong paper |
| Model phụ | — | Thêm NCL/DirectAU/SGL/SimGCL/LightGCL (port từ SELFRec), TAG-CF, PureMF | Mở rộng phạm vi so sánh SOTA |
| Cấu trúc thư mục | `data/`, 6 repo gốc rải ở root | `assets/{data,external_repos,papers}/` | Tổ chức lại cho dễ tái sử dụng/reproduce |
| Đóng gói | — | `Dockerfile` (PyTorch+CUDA 12.8), `docker-compose.yml` | Tái sử dụng môi trường dễ dàng |
| Script train | `run_pipeline.py`, `experiments/run_all.sh` | `train.sh` + `test.sh` (build bảng/biểu đồ cuối) | Gộp thành 1 entry point rõ ràng, có tag chọn model/kịch bản |
| Dữ liệu thiếu | Tự sinh synthetic (Zipf long-tail), gắn cờ `is_real_data` | Ném `FileNotFoundError` thẳng | Synthetic từng gây crash âm thầm khi trộn đơn vị timestamp; lỗi rõ ràng tốt hơn số liệu giả |

## Nguyên tắc chung

- **Mọi model theo API `BPRModelBase`** (`forward -> (user_emb,item_emb)`, `bpr_loss`,
  `getUsersRating`) để tái dùng `Procedure`/`utils` của LightGCN-PyTorch.
- Reproducibility: set seed qua `set_global_seed`, ghi config + `git HEAD` cạnh mỗi kết quả.
- Mọi kết quả trong `results/` luôn chạy trên real data (không có synthetic fallback).
- Mỗi model SOTA phải verify đối chiếu repo gốc của tác giả (vendor trong
  `assets/external_repos/`) trước khi đưa vào bảng so sánh chính.

## Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Global timestamp split gây cold-start (user không có train) | Đây là đặc tính chủ ý của split style, không phải bug — ghi rõ trong báo cáo |
| GPU dùng chung server, tốc độ dao động (thermal/tranh chấp) | Theo dõi `nvidia-smi` khi train dài; chấp nhận ETA co giãn |
| `torch-geometric` khó cài (GAT) | GAT optional (`--no-gat`); không nằm trong 5 model chuẩn |
| timestamp Amazon là ms, MovieLens là s | `AMAZON_GLOBAL_TS_T1/T2` hard-code riêng cho Amazon; dataset khác dùng `quantile_timestamp_cutoffs` tự suy từ đơn vị của chính nó — không trộn lẫn hằng số giữa 2 loại |
