# M1 — Problem framing & Data acquisition

Tương đương "Business Understanding + Data Generation" nhưng ở đây **không sinh dữ liệu**:
bài toán và dữ liệu đã cố định theo proposal; việc của phase này là nạp đúng dữ liệu thật.

## 1.1 Problem framing (đã có trong proposal — chỉ chốt lại)

- **Domain**: gợi ý sản phẩm e-commerce (beauty). Bài toán = **link prediction trên graph
  bipartite** user–item; đầu ra mỗi user = danh sách top-K item.
- **Task**: `ŷ_ui = f(u,i)`, `TopK(u) = argmax_i ŷ_ui`. Với Sheaf4Rec:
  `S = (F^u)^T F^v`, rank từng hàng.
- **Vì sao sheaf**: dữ liệu không có feature node giàu → chế độ collaborative filtering;
  Sheaf4Rec cải thiện *representation* (mỗi node mang một không gian vector + restriction map)
  thay vì phụ thuộc feature ngoài.

## 1.2 Đọc dữ liệu từ local (thay loader URL)

Sửa `gnn_recommendation/data.py` + thêm `DATA_ROOT` vào `config.py`:

```python
# config.py
from pathlib import Path
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
```

### Amazon All Beauty — 2 chế độ (`source`)

1. `source="raw"`: `pd.read_json(DATA_ROOT/"amazon_reviews_2023/raw/review_categories/All_Beauty.jsonl", lines=True)`,
   rename `parent_asin -> item_id`, giữ `[user_id, item_id, rating, timestamp]` (~700K review, tự split ở M3).
2. `source="benchmark"` (khuyến nghị cho số báo cáo): đọc sẵn 3 file
   `benchmark/<core>/<split>/All_Beauty.{train,valid,test}.csv`, `core ∈ {0core,5core}`,
   `split ∈ {last_out,timestamp}`. Có sẵn train/val/test → khớp "published reference" proposal.

> `timestamp` Amazon là **mili-giây** (13 chữ số) — dùng nhất quán, đừng trộn với giây.

### MovieLens 1M

`load_movielens_1m_real()` đọc `DATA_ROOT/"datamovielens-1m/ratings.dat"`
(`sep="::"`, engine python, latin-1, names `[user_id,item_id,rating,timestamp]`).
Giữ URL cũ chỉ như fallback cuối.

### Registry

`DATASET_REGISTRY`: `loader` trỏ file local; giữ `synthetic` làm fallback smoke test;
thêm khoá `benchmark_dir` cho amazon để pipeline chọn split dựng sẵn.

## Checklist M1

- [x] `config.py`: `DATA_ROOT`.
- [x] `data.py`: loader Amazon (raw + benchmark) + MovieLens local; cập nhật registry.
- [x] **Smoke**: `python run_pipeline.py --dataset amazon_beauty --epochs 1 --no-gat` cho `is_real_data=True`.
