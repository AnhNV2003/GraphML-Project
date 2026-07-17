# M1 — Problem framing & Data acquisition

Tương đương "Business Understanding + Data Generation" nhưng ở đây **không sinh dữ liệu**:
bài toán và dữ liệu đã cố định theo proposal; việc của phase này là nạp đúng dữ liệu thật.

## 1.1 Problem framing

- **Domain**: gợi ý sản phẩm e-commerce (Amazon Reviews 2023) + phim (MovieLens 1M, đối chứng).
  Bài toán = **link prediction trên graph bipartite** user–item; đầu ra mỗi user = danh sách
  top-K item.
- **Task**: `ŷ_ui = f(u,i)`, `TopK(u) = argmax_i ŷ_ui`. Với Sheaf4Rec:
  `S = (F^u)^T F^v`, rank từng hàng.
- **Vì sao sheaf**: dữ liệu không có feature node giàu → chế độ collaborative filtering;
  Sheaf4Rec cải thiện *representation* (mỗi node mang một không gian vector + restriction map)
  thay vì phụ thuộc feature ngoài.

## 1.2 Dataset registry (`gnn_recommendation/data.py`)

Hai dataset đăng ký trong `DATASET_REGISTRY`, tất cả đọc từ `assets/data/` (đường dẫn tuyệt
đối tính từ `gnn_recommendation/config.py:DATA_ROOT`, không hard-code máy cụ thể):

| Tên registry | Display name | Loader | Nguồn local |
|---|---|---|---|
| `amazon_video_games` | Amazon Video Games | `load_amazon_video_games_real` | `assets/data/amazon_reviews_2023/raw/review_categories/Video_Games.jsonl` |
| `movielens_1m` | MovieLens 1M | `load_movielens_1m_real` | `assets/data/datamovielens-1m/ratings.dat` |

Dataset Amazon dùng hàm tổng quát `load_amazon_category_real(category, ...)` — thêm category
mới chỉ cần đăng ký 1 wrapper + 1 entry trong `DATASET_REGISTRY`, không cần sửa logic load.
Amazon Video Games hiện là dataset Amazon duy nhất trong registry.

### Tải dữ liệu Amazon Reviews 2023

```bash
python install_dataset_huggingface.py --category Video_Games
```

`install_dataset_huggingface.py` dùng `huggingface_hub.snapshot_download` với
`local_dir=assets/data/amazon_reviews_2023`. **Lưu ý quan trọng**: `snapshot_download`
với `local_dir` coi thư mục local như một bản mirror và **xóa file không khớp
`allow_patterns`** ở mỗi lần gọi — script đã được sửa để tự động lấy union của mọi
category đã tải trước đó (`_already_downloaded_categories`) trước khi build
`allow_patterns`, để tải category mới không xóa mất category cũ.

- `source="raw"`: đọc trực tiếp `raw/review_categories/<Category>.jsonl`
  (`pd.read_json(lines=True)`), rename `parent_asin -> item_id`.
- `source="benchmark"`: đọc sẵn 3 file `benchmark/<core>/<split>/<Category>.{train,valid,test}.csv`
  (`core ∈ {0core,5core}`, `split ∈ {last_out,timestamp}`) — dùng cho smoke-test/đối chiếu,
  không phải đường chính (đường chính tự chạy `global_timestamp_split`, xem M3).

> `timestamp` Amazon là **mili-giây** (13 chữ số); MovieLens là **giây** (10 chữ số).
> Không trộn 2 đơn vị — xem cảnh báo trong `pipeline.py` (M3/M7).

### MovieLens 1M

`load_movielens_1m_real()` đọc local trước (`ratings.dat`, `sep="::"`, engine python,
latin-1, names `[user_id,item_id,rating,timestamp]`); nếu thiếu file, tự tải từ URL
GroupLens chính thức làm fallback cuối (`load_movielens_1m_from_url`).

### Không có synthetic fallback

`load_dataset_by_name()` gọi thẳng loader thật và **ném lỗi** (`FileNotFoundError`) nếu
dữ liệu local chưa tải — không còn cơ chế tự sinh dữ liệu giả để "chạy cho có". Toàn bộ
pipeline luôn chạy trên real data; nếu dataset chưa có, tải trước bằng
`install_dataset_huggingface.py --category <Tên>` (Amazon) hoặc để MovieLens tự tải từ
GroupLens (fallback URL duy nhất còn lại, vẫn là dữ liệu thật, chỉ khác nguồn local/mạng).

> **Lịch sử**: dự án từng có `generate_synthetic()` (Zipf-distributed long-tail) làm
> fallback khi thiếu dữ liệu thật, gắn cờ `is_real_data` để phân biệt. Đã **xóa hoàn toàn**
> khỏi code — cơ chế này từng gây edge case khó lường khi trộn đơn vị timestamp giữa dữ liệu
> synthetic và cutoff split thật. Bỏ synthetic fallback giúp lỗi thiếu dữ liệu hiện rõ ngay
> (`FileNotFoundError`) thay vì âm thầm chạy trên số liệu vô nghĩa.

## Checklist M1

- [x] `config.py`: `DATA_ROOT`, `ASSETS_ROOT`, `EXTERNAL_REPOS_ROOT` (tính từ `assets/`).
- [x] `data.py`: loader Amazon tổng quát theo category (raw + benchmark) + MovieLens local;
      registry gồm `amazon_video_games`, `movielens_1m`. Không còn synthetic
      fallback — `load_dataset_by_name` trả `DataFrame` trực tiếp, ném lỗi nếu thiếu data.
- [x] `install_dataset_huggingface.py`: hỗ trợ `--category`, không xóa nhầm category đã tải.
- [x] **Smoke**: `python -m experiments.exp_main --dataset amazon_video_games --epochs 1 --no-gat`
      chạy được trên dữ liệu thật.
