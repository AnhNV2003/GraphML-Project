# M2 — Exploratory Data Analysis

Dùng `gnn_recommendation/eda.py` (gọi qua `run_eda.py`) chạy trên **real data local** để lấy
số cho phần "Dataset" của báo cáo và biện minh các lựa chọn tiền xử lý ở M3.

## 2.1 Thống kê cần lấy

- `#users`, `#items`, `#interactions`, **sparsity** = 1 − |E|/(|U|·|I|).
- Phân bố **độ dài lịch sử** mỗi user và mỗi item (long-tail) → biện minh k-core.
- Phân bố **rating** (1–5) → biện minh ngưỡng implicit `rating ≥ 4` ở M3.
- Phân bố **timestamp** theo thời gian → biện minh global timestamp split (M3) và
  time-aware edges (M4).
- Trước/sau k-core: bảng so sánh #users/#items/#interactions để thấy tác động lọc.

## 2.2 Chạy

```bash
python run_eda.py --output-dir results
python run_eda.py --datasets amazon_video_games_raw,movielens_1m --benchmark-category Video_Games
```

Mặc định chạy trên `amazon_video_games_raw` + `movielens_1m` (đặt qua `--datasets`,
xem `RAW_SOURCES` trong `run_eda.py`). `--benchmark-category` (tùy chọn) thêm bảng so sánh
raw vs benchmark `0core`/`5core` timestamp cho 1 category Amazon cụ thể — gọi
`load_amazon_category_raw`/`load_amazon_category_benchmark` trực tiếp, không đi qua
`DATASET_REGISTRY`, vì EDA cần dữ liệu **trước khi** áp k-core/split để so sánh trước/sau.

## 2.3 Output

- `results/tables/dataset_statistics.csv` — thống kê cơ bản mỗi dataset.
- `results/tables/kcore_comparison.csv` — trước/sau k-core filter.
- `results/tables/amazon_benchmark_comparison.csv` — so sánh raw vs benchmark 0core/5core.
- `results/figures/{dataset}_history_lengths.png`, `_rating_distribution.png`,
  `_interactions_over_time.png`.
- `results/M2_eda_summary.md` — tổng hợp Markdown.

## 2.4 Lưu ý quan trọng

- Amazon Video Games rất thưa ở raw (2,766,656 users, median lịch sử user ~1). Với
  `MIN_INTERACTIONS=5` (k-core hiện tại), sau lọc còn 95,007 users / 25,838 items /
  816,455 interactions — đánh đổi có chủ ý giữa độ tin cậy thống kê (mỗi user/item có đủ
  tương tác) và kích thước dataset; xem M3 để biết con số cụ thể sau khi thêm lọc
  `rating≥4` và global timestamp split.

## Checklist M2

- [x] Chạy `run_eda.py` trên Amazon Video Games (raw + benchmark) + MovieLens, lưu số liệu + hình.
- [x] Bảng thống kê dataset (trước/sau k-core).
- [x] Chốt cấu hình dữ liệu (raw + k-core=5) dựa trên EDA.
