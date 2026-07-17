# M3 — Preprocessing (cleaning + implicit feedback + split)

Tương đương "Data Cleaning". Code trong `gnn_recommendation/preprocessing.py`.

## 3.1 Pipeline tiền xử lý (`preprocess()`)

Thứ tự thực hiện (đúng thứ tự code chạy):

1. **Implicit feedback threshold**: `rating >= positive_threshold` (mặc định `4.0`,
   truyền `None` để giữ mọi rating — dùng khi thử nghiệm rating-aware edges ở M4).
2. **Dedup**: `drop_duplicates(["user_id","item_id"], keep="last")`.
3. **K-core lặp** (tối đa 5 vòng, tới khi hội tụ): loại user/item có ít hơn
   `MIN_INTERACTIONS` tương tác, lặp lại vì loại item có thể khiến user tụt dưới
   ngưỡng và ngược lại.
4. Map `u_idx`, `i_idx` liên tục (0-indexed) cho toàn bộ user/item còn lại.

```python
if positive_threshold is not None and "rating" in data.columns:
    data = data[data["rating"] >= positive_threshold]
# ... dedup ...
for _ in range(max_iter):
    valid_users = counts_user[counts_user >= min_interactions].index
    valid_items = counts_item[counts_item >= min_interactions].index
    new_data = data[data["user_id"].isin(valid_users) & data["item_id"].isin(valid_items)]
    if len(new_data) == len(data):
        break
    data = new_data
```

## 3.2 K-core threshold: `MIN_INTERACTIONS = 5`

Cấu hình tại `gnn_recommendation/config.py:MIN_INTERACTIONS`. Giá trị này **đã tăng từ 2
lên 5** so với giai đoạn đầu dự án — lọc mạnh hơn để mỗi user/item còn lại có đủ tín hiệu
tương tác, đánh đổi lấy kích thước dataset nhỏ hơn (đặc biệt rõ trên Amazon Beauty vốn đã
rất thưa — xem số liệu k-core trong `results/tables/kcore_comparison.csv`, sinh từ M2).

## 3.3 Split: **global timestamp split** (đường chính, duy nhất)

> **Đã đổi hoàn toàn so với giai đoạn đầu dự án.** Chiến lược leave-one-out (per-user
> relative cut, mỗi user luôn có ≥1 tương tác test) đã được thử nghiệm ban đầu nhưng
> **bị xóa hẳn khỏi code** — không còn hàm `leave_one_out_split` trong
> `preprocessing.py`. Toàn bộ pipeline giờ chỉ dùng 1 giao thức split duy nhất.

`global_timestamp_split(data, t1, t2)`: một cặp cutoff **tuyệt đối** áp dụng cho toàn
dataset cùng lúc (khác LOO — không phải per-user):

```python
def global_timestamp_split(data, t1, t2):
    train_rows = data[data["timestamp"] < t1]
    valid_rows = data[(data["timestamp"] >= t1) & (data["timestamp"] < t2)]
    test_rows  = data[data["timestamp"] >= t2]
    return train_rows.reset_index(drop=True), pairs_from_frame(valid_rows), pairs_from_frame(test_rows)
```

Đây là phong cách split chính thức của benchmark **Amazon Reviews 2023** ("timestamp"
split). Tính chất quan trọng: **user có toàn bộ lịch sử rơi sau `t1` sẽ có 0 tương tác
train** — bị đánh giá hoàn toàn trên embedding khởi tạo ngẫu nhiên (cold-start). Đây là
đặc tính chủ ý của giao thức, không phải bug — và là lý do quan sát thấy **test luôn có
NDCG@10 thấp hơn valid** ở mọi model (test set có tỷ lệ cold-start cao hơn valid set, vì
`t2` cắt muộn hơn).

### Xác định `(t1, t2)` cho từng dataset

- **`amazon_beauty`**: dùng `AMAZON_GLOBAL_TS_T1`/`AMAZON_GLOBAL_TS_T2`
  (`gnn_recommendation/pipeline.py`) — hằng số **ms-epoch** reverse-engineer trực tiếp từ
  file benchmark gốc `assets/data/amazon_reviews_2023/benchmark/0core/timestamp/*.csv`,
  khớp chính xác cutoff chính thức của Amazon Reviews 2023.
- **Mọi dataset khác** (`amazon_video_games`, `movielens_1m`, ...): dùng
  `quantile_timestamp_cutoffs(data)` — tự suy `(t1, t2)` từ phân vị (84.05% / 10.34% /
  5.61%, đúng tỷ lệ train/valid/test của benchmark gốc) trên timestamp của chính dataset
  đó, để tỷ lệ split nhất quán dù đơn vị/khoảng thời gian tuyệt đối khác nhau.

> **Cảnh báo đơn vị timestamp**: `AMAZON_GLOBAL_TS_T1/T2` là hằng số **ms-epoch**, chỉ
> đúng cho Amazon. Không được tái sử dụng trực tiếp cho dataset khác dù cùng họ Amazon —
> mỗi dataset mới nên tự tính qua `quantile_timestamp_cutoffs`, trừ khi đã verify đơn vị
> timestamp khớp chính xác với Amazon Beauty gốc.

### `GraphRecDataset`

`gnn_recommendation/official.py:GraphRecDataset.create` nhận cả `valid_pairs` và
`test_pairs`, expose `validDict` song song `testDict` — dùng cho model selection/early
stopping (M6) mà không cần sửa `BasicDataset` gốc của LightGCN-PyTorch.

## Checklist M3

- [x] `positive_threshold` (mặc định 4.0, cho phép None).
- [x] K-core lặp, `MIN_INTERACTIONS=5`.
- [x] `global_timestamp_split` — đường chính, duy nhất (LOO đã xóa khỏi code).
- [x] `quantile_timestamp_cutoffs` cho dataset không có cutoff chính thức hard-code.
- [x] Train giữ `rating`/`timestamp` cho M4 (edge weighting).
- [x] `GraphRecDataset`: `validDict` song song `testDict`.
