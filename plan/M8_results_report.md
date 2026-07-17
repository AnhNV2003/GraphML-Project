# M8 — Results & Report

Đóng gói kết quả thành bảng + hình cho báo cáo, đóng gói môi trường qua Docker để dễ
tái sử dụng.

## 8.1 Chạy full

```bash
bash train.sh              # toàn bộ: E0-E4 + SOTA + TAG-CF + PureMF, cả 2 dataset, 3 seed
bash train.sh e0            # chỉ 1 giai đoạn — xem `bash train.sh` (không tham số) để rõ danh sách
```

- Mọi kết quả luôn chạy trên real data (không có synthetic fallback — thiếu dữ liệu sẽ
  báo lỗi thẳng thay vì chạy tiếp với số liệu giả). Log ghi vào `results/logs/<stage>.log`.
- Có thể override `MODELS=`, `DATASET_MAIN=`, `DATASET_SWEEP=`, `SEEDS=`,
  `EPOCHS_MAIN=`, `EPOCHS_SWEEP=`, `PATIENCE=` qua biến môi trường (xem đầu `train.sh`).

## 8.2 Tổng hợp báo cáo (`test.sh` + `experiments/build_report.py`)

```bash
bash test.sh                # build bảng + biểu đồ từ results/*.csv hiện có
bash test.sh --smoke        # chạy nhanh (--quick) toàn bộ 8 script trước, rồi build report
```

`build_report.py` đọc `main_comparison.csv`, `sota_comparison.csv`, `tagcf_comparison.csv`,
`puremf_comparison.csv` (chuẩn hóa cột `Recall@10`/`NDCG@10` giữa 2 định dạng khác nhau —
một số script ghi `_mean` suffix qua `summarize_mean_std`, `exp_tagcf.py` ghi cột thô), gộp
thành bảng xếp hạng theo từng dataset, cộng với `layer_sweep.csv`/`dim_sweep.csv`/
`expressiveness.csv`/`edge_construction.csv` cho các hình sweep (bỏ qua nhẹ nhàng nếu file
chưa tồn tại — không lỗi cứng).

### Output

- `results/M8_comparison.md` — bảng xếp hạng gộp (Recall@10/NDCG@10) theo từng dataset,
  kèm huy chương 🥇🥈🥉 cho top-3.
- `results/tables/model_comparison_<dataset>.md` — bảng riêng từng dataset.
- `results/figures/model_comparison_<dataset>.png` — bar chart Recall@10 & NDCG@10.
- `results/figures/e1_layer_sweep.png`, `e2_dim_sweep_latent.png`, `e2_dim_sweep_sheaf.png`,
  `e3_expressiveness.png`, `e4_edge_construction.png` — line/bar chart sweep (chỉ sinh nếu
  CSV tương ứng tồn tại).

## 8.3 Code snippet cho báo cáo viết tay (LaTeX)

`plan/report_code_snippets.tex` chứa 5 đoạn code đã trích sẵn (định dạng `lstlisting`) để
dán vào phần "Appropriateness & explanation of model(s)" của báo cáo:

1. LightGCN propagation (baseline quy chiếu).
2. Sheaf4Rec restriction map + sheaf diffusion (đóng góp lý thuyết trung tâm).
3. NGCF forward (element-wise interaction + concat readout).
4. DirectAU alignment+uniformity loss (model duy nhất không dùng BPR).
5. Global timestamp split (giao thức đánh giá, giải thích vì sao test luôn thấp hơn valid).

Dùng `\input{plan/report_code_snippets}` hoặc copy tay từng `lstlisting` vào report chính.

## 8.4 Đóng gói Docker

```bash
docker compose build
docker compose run --rm gnn-recommendation bash train.sh e0
```

- `Dockerfile`: base image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` (khớp
  `torch==2.8.0+cu128` trong `requirements.txt`), không cần `apt-get` — repo vendor
  (`assets/external_repos/`) đã bake sẵn vào image nên `git` không cần thiết ở runtime;
  `torch-geometric`'s `GATConv` chạy được với build pip thuần, không cần biên dịch
  (`gcc`/`build-essential`).
- `docker-compose.yml`: mount `./assets/data` và `./results` ra ngoài container (dataset
  nặng, kết quả cần giữ lại sau khi container bị xóa), cấu hình GPU reservation
  (`nvidia-container-toolkit`).
- `.dockerignore`: loại `venv/`, `assets/data/` khỏi build context.
- Đã verify: build sạch, `torch.cuda.is_available()=True` trong container, network egress
  hoạt động (MovieLens tự tải fallback khi thiếu volume mount), pipeline train thật chạy
  đúng end-to-end.

## 8.5 Viết mục Results/Analysis (theo cấu trúc proposal §5)

- **Outcome**: bảng E0 — so sánh 5 model chuẩn trên Recall@10, NDCG@10, F1, MRR; bàn về
  cân bằng precision–recall; giải thích test luôn thấp hơn valid (cold-start do global
  timestamp split, xem M3).
- **Mechanism**: layer sweep (E1, over-smoothing), accuracy–cost (E2), Sheaf4Rec
  expressiveness (E3, kiểm chứng cải thiện đến từ cấu trúc không chỉ từ #tham số), edge
  construction (E4), timing (M6, "slower to train, faster to serve").

## 8.6 Tài liệu đã cập nhật

- `README.md` — cấu trúc repo, cách chạy `train.sh`/`test.sh`/Docker, split protocol.
- `DATASET_COLUMNS.md` — schema thực tế, path `assets/data/`.
- `requirements.txt` — pin version (`torch`, `torch-geometric`, `huggingface_hub`...), xóa
  dependency chết (`seaborn`, `datasets` không dùng ở đâu).
- `plan/` (toàn bộ M1-M8 + README) — đồng bộ với trạng thái code hiện tại.

## Checklist M8

- [x] `train.sh`/`test.sh` chạy được độc lập từng giai đoạn.
- [x] `build_report.py` gộp bảng + vẽ hình từ mọi CSV có sẵn.
- [x] `plan/report_code_snippets.tex` — 5 snippet cho báo cáo viết tay.
- [x] Docker: build + chạy thử thành công (GPU, network egress).
- [x] README/DATASET_COLUMNS/requirements.txt đồng bộ code.
- [ ] Full run `train.sh all` trên Amazon Video Games + LOO cũ (đang tiến hành — theo dõi
      `results/logs/video_games_full.log`).
- [ ] Viết mục Results/Analysis (outcome + mechanism) trong report chính thức.
