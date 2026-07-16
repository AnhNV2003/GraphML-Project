# Sheaf Neural Networks cho E-Commerce Recommendation

So sánh 5 model đồ thị chuẩn cho bài toán recommendation trên Amazon Reviews 2023 + MovieLens 1M:
**LightGCN** (SIGIR'20), **NGCF** (SIGIR'19), **Sheaf4Rec** (ACM TORS'23/25), **NCL** (WWW'22), **DirectAU** (KDD'22).

Mỗi model được port trung thành từ repo gốc của tác giả (xem `assets/external_repos/`), không tự ý đơn giản hóa kiến trúc, để đảm bảo kết quả so sánh công bằng.

## Cấu trúc repo

```
gnn_recommendation/     # Package lõi: data loading, preprocessing, graph, model, training, pipeline
experiments/             # Entry-point scripts (mỗi model/kịch bản train qua đây)
assets/
  data/                   # Dataset local (Amazon Reviews 2023, MovieLens 1M)
  external_repos/         # 6 repo gốc vendor (tham chiếu để verify tính trung thành của port)
  papers/                 # PDF tham khảo
  smoke_tests/            # Output CSV của các lần smoke-test cũ
results/                  # CSV/figures/tables — output của train.sh/test.sh
plan/                     # Tài liệu kế hoạch M1-M8
train.sh, test.sh         # Script train + tổng hợp báo cáo
```

### `gnn_recommendation/` (package lõi)

| File | Vai trò |
|---|---|
| `config.py` | Seed, device, hyperparameter mặc định, đường dẫn `assets/` |
| `data.py` | Loader Amazon Reviews 2023 (theo category), MovieLens 1M, synthetic fallback |
| `preprocessing.py` | Lọc positive-feedback, k-core filtering, split time-based (global timestamp) |
| `graph.py` | Dựng bipartite graph chuẩn hóa dạng sparse tensor |
| `official.py` | Clone/import `LightGCN-PyTorch` gốc, stub module `world`, adapter dataset |
| `model_base.py` | Interface `BPRModelBase` dùng chung cho mọi model |
| `extra_models.py` | NGCF, GAT, Popularity, MF-stub |
| `sheaf.py` / `sheaf_official.py` | Sheaf4Rec — bản tự viết và bản port trung thành từ repo gốc |
| `ssl_models.py` | NCL, DirectAU, SGL, SimGCL, LightGCL (port từ SELFRec) |
| `tagcf.py` | MF + TAG-CF (test-time message passing) |
| `training.py` | Train loop BPR + tính metric (Recall/NDCG/Precision/F1/MRR/HitRatio) |
| `pipeline.py` | Ghép toàn bộ pipeline: load → preprocess → split → graph → train → eval |

### `experiments/` (entry points)

| File | Kịch bản |
|---|---|
| `exp_main.py` | **E0** — bảng so sánh chính giữa 5 model |
| `exp_layers.py` | **E1** — sweep số lớp truyền tin (n_layers) |
| `exp_dim.py` | **E2** — sweep embedding dim (+ stalk dim riêng cho Sheaf4Rec) |
| `exp_expressiveness.py` | **E3** — so sánh 3 kiểu restriction map của Sheaf4Rec |
| `exp_edges.py` | **E4** — so sánh cách tính trọng số cạnh (binary/rating/time) |
| `exp_sota.py` | Train 5 model self-supervised phụ (SGL, SimGCL, DirectAU, NCL, LightGCL) |
| `exp_tagcf.py` | Train MF + TAG-CF |
| `exp_puremf.py` | Train PureMF (baseline từ LightGCN-PyTorch gốc) |
| `exp_tune_official.py` | Grid-search hyperparameter cho Sheaf4Rec-official |
| `build_report.py` | Gộp mọi CSV kết quả thành bảng + biểu đồ cuối |

## Cài đặt

### Cách 1: venv trực tiếp

```bash
python3 -m venv venv
# Cài torch đúng CUDA trước (xem comment đầu requirements.txt cho version phù hợp GPU của bạn)
venv/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
venv/bin/pip install -r requirements.txt
```

### Cách 2: Docker (khuyến nghị để tái sử dụng)

```bash
docker compose build
docker compose run --rm gnn-recommendation bash train.sh e0
```

Xem chi tiết ở mục [Docker](#docker) bên dưới.

## Dữ liệu

Dataset thật đọc từ `assets/data/`:

- **Amazon Reviews 2023**: `assets/data/amazon_reviews_2023/raw/review_categories/<Category>.jsonl`. Tải bằng:
  ```bash
  python install_dataset_huggingface.py --category All_Beauty
  python install_dataset_huggingface.py --category Video_Games
  ```
  (script tự giữ lại category đã tải trước đó, không ghi đè khi tải category mới)
- **MovieLens 1M**: `assets/data/datamovielens-1m/ratings.dat` — tự tải từ GroupLens nếu thiếu file local.

Nếu dữ liệu thật không đọc được, pipeline dùng synthetic fallback (chỉ để kiểm tra pipeline chạy đúng luồng, không dùng làm số liệu báo cáo).

## Split protocol

Toàn bộ pipeline dùng **global timestamp split** (theo phong cách benchmark chính thức của Amazon Reviews 2023): một cặp cutoff `(t1, t2)` áp dụng cho toàn dataset — `train = t < t1`, `valid = t1 ≤ t < t2`, `test = t ≥ t2`. Khác với leave-one-out (mỗi user luôn có ít nhất 1 tương tác test), split này có cold-start tự nhiên: user có toàn bộ lịch sử sau `t1` sẽ có 0 tương tác train.

## Chạy training

```bash
bash train.sh              # train toàn bộ: E0-E4 + SOTA + TAG-CF + PureMF, cả 2 dataset
bash train.sh e0            # chỉ E0 (bảng so sánh chính)
bash train.sh e1            # chỉ E1 (layer sweep)
# ... e2, e3, e4, sota, tagcf, puremf tương tự

# Tùy biến qua biến môi trường:
MODELS="LightGCN,NGCF" bash train.sh e0
DATASET_MAIN=amazon_beauty bash train.sh e0
SEEDS=42 EPOCHS_MAIN=20 bash train.sh e0
```

Log ghi vào `results/logs/<stage>.log`, kết quả CSV ghi vào `results/`.

## Tổng hợp báo cáo

```bash
bash test.sh                # build bảng + biểu đồ từ results/*.csv hiện có
bash test.sh --smoke        # chạy nhanh (--quick, ~1 epoch/model) rồi build report — dùng để kiểm tra pipeline không lỗi
```

Output:
- `results/M8_comparison.md` — bảng xếp hạng gộp (Recall@10/NDCG@10) theo từng dataset
- `results/tables/model_comparison_*.md`
- `results/figures/model_comparison_*.png`, `e1_layer_sweep.png`, `e2_dim_sweep_*.png`, `e3_expressiveness.png`, `e4_edge_construction.png`

## Docker

`Dockerfile` dùng base image `pytorch/pytorch` (đã có CUDA 12.8 + torch GPU sẵn), cài thêm `torch-geometric` + các dependency còn lại.

```bash
# Build image
docker compose build

# Chạy training (cần nvidia-container-toolkit để dùng GPU)
docker compose run --rm gnn-recommendation bash train.sh e0

# Chạy shell tương tác để debug
docker compose run --rm gnn-recommendation bash
```

`docker-compose.yml` mount 2 volume ra ngoài container để dữ liệu/kết quả không mất khi container bị xóa:
- `./assets/data` — dataset (không bake vào image vì khá nặng, ~4-5GB)
- `./results` — output training

`assets/external_repos/` (6 repo gốc vendor) được bake sẵn vào image ở build-time.

Yêu cầu máy host: **GPU NVIDIA + driver hỗ trợ CUDA 12.8 + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**. Không có GPU vẫn chạy được (tự fallback CPU) nhưng chậm hơn nhiều, chỉ nên dùng để smoke-test.

## Ghi chú

- `assets/external_repos/` chứa 6 repo gốc (LightGCN-PyTorch, NGCF-PyTorch-official, Sheaf4Rec-official, SELFRec-official, TAG-CF-official, LightGCL-official) dùng làm tài liệu tham chiếu khi verify tính trung thành của các port trong `gnn_recommendation/`. Chỉ `LightGCN-PyTorch` được import trực tiếp lúc chạy (qua `official.py`); các repo còn lại chỉ dùng để đối chiếu code khi viết lại.
- `plan/` chứa tài liệu kế hoạch M1-M8 gốc của dự án.
