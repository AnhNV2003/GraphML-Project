# Dataset Columns

Project hiện dùng dữ liệu local trong `assets/data/` cho 2 dataset:

- `amazon_beauty`: Amazon Reviews 2023, category All Beauty.
- `movielens_1m`: MovieLens 1M.

Pipeline chuẩn hoá cả 2 dataset về schema interaction tối thiểu:

```text
user_id, item_id, rating, timestamp
```

Sau preprocessing, pipeline thêm:

```text
u_idx, i_idx
```

## 1. Amazon Beauty

Loader chính: `load_amazon_beauty_real()` trong `gnn_recommendation/data.py`.

Nguồn local:

```text
assets/data/amazon_reviews_2023/
```

Mặc định hiện tại của project sau M2 là `source="raw"` để giữ dữ liệu đầy đủ rồi lọc ở preprocessing/M3. Benchmark vẫn có sẵn cho smoke test hoặc thí nghiệm reference-aligned.

Benchmark `source="benchmark"`, `core="5core"`, `split="timestamp"` đọc:

```text
assets/data/amazon_reviews_2023/benchmark/5core/timestamp/All_Beauty.train.csv
assets/data/amazon_reviews_2023/benchmark/5core/timestamp/All_Beauty.valid.csv
assets/data/amazon_reviews_2023/benchmark/5core/timestamp/All_Beauty.test.csv
```

### 1.1 Amazon benchmark files

Các file benchmark dạng thường (`last_out`, `timestamp`) có header:

```text
user_id,parent_asin,rating,timestamp
```

Các file benchmark dạng `_w_his` (`last_out_w_his`, `timestamp_w_his`) có thêm:

```text
history
```

Pipeline hiện mặc định dùng `5core/timestamp`, nên không dùng cột `history`.

| Cột trên disk | Cột trong pipeline | Kiểu dự kiến | Mô tả |
|---|---|---:|---|
| `user_id` | `user_id` | string | ID ẩn danh của user. Ví dụ mẫu: `AFSKPY37N3C43SOI5IEXEK5JSIYA`. |
| `parent_asin` | `item_id` | string | ID sản phẩm cấp parent ASIN. Code rename `parent_asin -> item_id`. |
| `rating` | `rating` | float | Điểm review, thường 1-5. Pipeline dùng cặp user-item như implicit feedback, không dự đoán rating trực tiếp. |
| `timestamp` | `timestamp` | int | Thời điểm review theo mili-giây Unix timestamp, 13 chữ số. Không trộn với timestamp giây của MovieLens. |
| `history` | chưa dùng | string/empty | Chỉ có ở các split `_w_his`; thường là chuỗi item trước đó của user. Pipeline mặc định không đọc split này. |

Các nhánh benchmark local hiện có:

```text
benchmark/{0core,5core}/{last_out,last_out_w_his,timestamp,timestamp_w_his}/
```

Ngoài ra có:

```text
benchmark/{0core,5core}/rating_only/All_Beauty.csv
```

File `rating_only` có cùng cột cơ bản `user_id,parent_asin,rating,timestamp`, nhưng không phải split train/valid/test.

### 1.2 Amazon raw review file

Raw review file:

```text
assets/data/amazon_reviews_2023/raw/review_categories/All_Beauty.jsonl
```

Mỗi dòng là một JSON review. Mẫu đầu cho thấy các cột:

| Cột raw | Pipeline dùng? | Kiểu dự kiến | Mô tả |
|---|---:|---:|---|
| `rating` | có | float | Điểm review. |
| `title` | không | string | Tiêu đề review. |
| `text` | không | string | Nội dung review. |
| `images` | không | list | Ảnh gắn với review, nếu có. |
| `asin` | không | string | ASIN biến thể/sản phẩm cụ thể. |
| `parent_asin` | có, rename thành `item_id` | string | Parent ASIN dùng làm item ID trong pipeline. |
| `user_id` | có | string | ID ẩn danh của user. |
| `timestamp` | có | int | Unix timestamp mili-giây. |
| `helpful_vote` | không | int | Số lượt helpful vote. |
| `verified_purchase` | không | bool | Review có phải verified purchase hay không. |

### 1.3 Amazon metadata file

Metadata local:

```text
assets/data/amazon_reviews_2023/raw/meta_categories/meta_All_Beauty.jsonl
```

Pipeline hiện chưa dùng file này, nhưng có thể dùng ở phase sau để thêm side information.

Các cột thấy trong mẫu đầu:

| Cột metadata | Kiểu dự kiến | Mô tả |
|---|---:|---|
| `main_category` | string | Category chính, ví dụ `All Beauty`. |
| `title` | string | Tên sản phẩm. |
| `average_rating` | float | Rating trung bình của sản phẩm. |
| `rating_number` | int | Số lượng rating/review. |
| `features` | list | Danh sách feature text. |
| `description` | list | Mô tả sản phẩm. |
| `price` | float/null/string | Giá, có thể null hoặc không chuẩn hoá. |
| `images` | list[dict] | Ảnh sản phẩm với các URL như `thumb`, `large`, `hi_res`. |
| `videos` | list | Video sản phẩm nếu có. |
| `store` | string | Tên brand/store. |
| `categories` | list | Cây category phụ nếu có. |
| `details` | dict | Thuộc tính sản phẩm, ví dụ kích thước, UPC. |
| `parent_asin` | string | Khoá join tiềm năng với interaction `parent_asin/item_id`. |
| `bought_together` | list/null | Sản phẩm thường được mua cùng, nếu có. |

## 2. MovieLens 1M

Loader chính: `load_movielens_1m_real()` trong `gnn_recommendation/data.py`.

Nguồn local:

```text
assets/data/datamovielens-1m/
```

Pipeline hiện chỉ dùng `ratings.dat`; `users.dat` và `movies.dat` có sẵn nhưng chưa dùng trong model hiện tại.

### 2.1 ratings.dat

File:

```text
assets/data/datamovielens-1m/ratings.dat
```

Format gốc:

```text
UserID::MovieID::Rating::Timestamp
```

Mapping vào pipeline:

| Cột gốc | Cột trong pipeline | Kiểu dự kiến | Mô tả |
|---|---|---:|---|
| `UserID` | `user_id` | string | ID user MovieLens. Code đọc rồi chuẩn hoá thành string. |
| `MovieID` | `item_id` | string | ID phim/item. Code đọc rồi chuẩn hoá thành string. |
| `Rating` | `rating` | int | Rating 1-5. Pipeline dùng làm interaction implicit. |
| `Timestamp` | `timestamp` | int | Unix timestamp theo giây, 10 chữ số. Khác Amazon, Amazon là mili-giây. |

### 2.2 users.dat

File:

```text
assets/data/datamovielens-1m/users.dat
```

Format gốc:

```text
UserID::Gender::Age::Occupation::Zip-code
```

Pipeline hiện chưa dùng file này.

| Cột gốc | Kiểu dự kiến | Mô tả |
|---|---:|---|
| `UserID` | int/string | ID user, join được với `ratings.dat.UserID`. |
| `Gender` | string | Giới tính mã hoá, ví dụ `F`, `M`. |
| `Age` | int | Nhóm tuổi mã hoá theo MovieLens. |
| `Occupation` | int | Nghề nghiệp mã hoá bằng ID. |
| `Zip-code` | string | ZIP code của user. |

### 2.3 movies.dat

File:

```text
assets/data/datamovielens-1m/movies.dat
```

Format gốc:

```text
MovieID::Title::Genres
```

Pipeline hiện chưa dùng file này.

| Cột gốc | Kiểu dự kiến | Mô tả |
|---|---:|---|
| `MovieID` | int/string | ID phim, join được với `ratings.dat.MovieID`. |
| `Title` | string | Tên phim, thường kèm năm trong ngoặc. |
| `Genres` | string | Danh sách genre nối bằng dấu `|`, ví dụ `Animation|Children's|Comedy`. |

## 3. Cột sinh ra sau preprocessing

Sau khi loader trả schema tối thiểu, `preprocess()` mặc định giữ implicit positive feedback `rating >= 4.0`, chạy k-core với `MIN_INTERACTIONS=2`, rồi sinh thêm:

| Cột | Kiểu | Mô tả | Ghi chú |
|---|---:|---|---|
| `u_idx` | int | Chỉ số user liên tục từ `0` đến `n_users - 1`. | Sinh từ `user_id` sau deduplicate và k-core filtering. |
| `i_idx` | int | Chỉ số item liên tục từ `0` đến `n_items - 1`. | Sinh từ `item_id` sau deduplicate và k-core filtering. |

## 4. Các cấu trúc trung gian trong pipeline

Các cột trên được chuyển thành những cấu trúc sau để train/evaluate model:

| Tên | Kiểu | Mô tả |
|---|---:|---|
| `train_pairs` | `list[tuple[int, int]]` | Danh sách cặp `(u_idx, i_idx)` dùng để train. |
| `valid_pairs` | `list[tuple[int, int]]` | Danh sách validation pairs; chỉ user có ít nhất 3 interaction sau lọc mới có validation item. |
| `test_pairs` | `list[tuple[int, int]]` | Danh sách cặp `(u_idx, i_idx)` dùng để test, mỗi user có 1 item test theo leave-one-out hiện tại của pipeline. |
| `Graph` | `torch.sparse_coo_tensor` | Ma trận kề bipartite user-item đã symmetric hoá và normalized. |
| `allPos` | `list[np.ndarray]` | Với mỗi user index, chứa các item index đã xuất hiện trong train. Dùng để negative sampling và loại item train khi evaluate. |
| `validDict` | `dict[int, list[int]]` | Mapping user index sang item validation, phục vụ model selection ở phase sau. |
| `testDict` | `dict[int, list[int]]` | Mapping user index sang item test. Dùng cho Recall, Precision, NDCG và HitRatio. |

`Graph` hỗ trợ 3 chế độ trọng số cạnh:

| `edge_mode` | Trọng số trước normalize | Ghi chú |
|---|---|---|
| `binary` | `1.0` cho mọi interaction train | Mặc định, baseline LightGCN/PureMF-compatible. |
| `rating` | `rating / 5.0` | Nên chạy với `--positive-threshold none` nếu muốn rating còn mang thông tin 1-5. |
| `time` | `exp(-(1 - recency))`, với recency chuẩn hoá theo timestamp train | Timestamp Amazon là mili-giây, MovieLens là giây; công thức dùng min/max nên không trộn đơn vị giữa dataset. |

Sau M5, các model sheaf dùng graph này để dựng sparse block sheaf Laplacian. `Sheaf4Rec-gcn_like` dùng scalar restriction map, `Sheaf4Rec-gat_like` dùng scalar attention-like restriction map, và `Sheaf4Rec-full_sheaf` dùng restriction map đầy đủ `d x d`.

## 5. Synthetic fallback

Nếu local file bị thiếu hoặc loader không đọc được dữ liệu thật, project tự sinh synthetic fallback với cùng schema:

```text
user_id, item_id, rating, timestamp
```

| Cột | Mô tả |
|---|---|
| `user_id` | ID giả dạng `U{number}`. |
| `item_id` | ID giả dạng `I{number}`. |
| `rating` | Rating giả trong thang 1-5. |
| `timestamp` | Timestamp giả trong khoảng Unix timestamp cố định. |

Kết quả chạy trên synthetic fallback chỉ dùng để kiểm tra pipeline, không nên dùng làm số liệu báo cáo.
