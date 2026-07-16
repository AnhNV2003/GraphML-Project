# M3 — Preprocessing (cleaning + implicit feedback + split)

Tương đương "Data Cleaning". Sửa `gnn_recommendation/preprocessing.py`.

## 3.1 Đã có (giữ)

- `drop_duplicates(["user_id","item_id"], keep="last")`.
- k-core lặp (`min_interactions`) cho cả user và item.
- Map `u_idx`, `i_idx` liên tục.

## 3.2 Implicit feedback threshold (proposal: `R_ui = 1 ⇔ rating ≥ 4`)

Thêm tham số `positive_threshold: float | None = 4.0` vào `preprocess(...)`, áp **trước**
dedup và k-core:

```python
if positive_threshold is not None and "rating" in data.columns:
    data = data[data["rating"] >= positive_threshold]
```

Cho phép `None` để giữ mọi tương tác khi làm thí nghiệm rating-aware ở M4/M7. Log số dòng
trước/sau.

## 3.3 Split train / validation / test

> **Ràng buộc từ M2 (đã đo, quan trọng):** Amazon All Beauty cực thưa (median lịch sử user = 1).
> Thứ tự thực tế `rating≥4 → k-core` làm dữ liệu sụp nhanh:
> raw 701K → rating≥4 500K → **k-core=2: 33K/12.6K user → k-core=3: chỉ 6.7K/1.0K user**.
> Ở k-core=2 chỉ **19.5% user có ≥3 tương tác**. Vì vậy **KHÔNG ép `MIN_INTERACTIONS=3`** cho
> Amazon (sẽ giết dataset), và **KHÔNG** thể làm per-user leave-one-out 3 phần cho mọi user.

Split tolerant với lịch sử ngắn: **test luôn per-user leave-one-out**; **val chỉ trích cho user
đủ dài** (≥3), user ngắn chỉ đóng góp train (không có val):

```python
def leave_one_out_split(data, val=True, min_for_val=3):
    data = data.sort_values(["u_idx", "timestamp"])
    test = data.groupby("u_idx").tail(1)          # mọi user >=2 đều có test
    rest = data.drop(test.index)
    if val:
        sizes = data.groupby("u_idx").size()
        eligible = sizes[sizes >= min_for_val].index
        valid = rest[rest["u_idx"].isin(eligible)].groupby("u_idx").tail(1)
        train = rest.drop(valid.index)
        return train_df(train), pairs(valid), pairs(test)   # train giữ DataFrame cho M4
    return train_df(rest), pairs(test)
```

- **Giữ `MIN_INTERACTIONS = 2`** (config hiện tại đã đúng — không đổi thành 3). MovieLens dày
  (mọi user ≥20) nên val per-user chạy tốt; Amazon dùng cơ chế val-có-điều-kiện ở trên. Val set
  Amazon ~2.5k user — đủ cho model selection ở M6.
- Cân nhắc thêm: nếu muốn val phủ nhiều user hơn, dùng global random holdout X% tương tác train
  làm val thay vì leave-one-out. Chốt 1 cách, ghi rõ trong báo cáo, dùng nhất quán mọi model.
- **Đường chính (đồng nhất cho cả 2 dataset)**: dùng self-split leave-one-out 3 phần này.
  MovieLens không có benchmark split nên đây là cách duy nhất so sánh nhất quán. Loader Amazon
  của M1 đã **gộp phẳng** benchmark (bỏ cột `benchmark_split`) → khớp thẳng với đường này; hệ quả:
  lựa chọn `benchmark_split` (`timestamp`/`last_out`) và các khoá metadata registry hiện **không
  ảnh hưởng** đường chính (đã flag ở verify M1).
- **Đường phụ (tùy chọn, sanity-check vs published reference)**: dùng thẳng 3 file CSV benchmark
  của Amazon, bỏ qua hàm split. Cần bổ sung một loader **không gộp phẳng** (giữ `benchmark_split`)
  — M1 chưa có, làm khi thực sự cần đối chiếu số công bố; không bắt buộc cho đường chính.
- **Giữ cột `rating`, `timestamp` trong train** (không chỉ `(u_idx,i_idx)`) để M4 dựng được
  edge rating/time-aware. Đề xuất: trả DataFrame train + list pairs cho val/test.
  (M1 đã giữ `rating`/`timestamp` ở loader → OK.)
- Cập nhật `official.GraphRecDataset.create` nhận `valid_pairs` và expose `validDict` song song
  `testDict` (phục vụ model selection ở M6).
- **Chốt core level trước khi split** (dựa trên M2 EDA): 5core chỉ ~2.5k tương tác/253 user →
  quá nhỏ, số nhiễu. Cân nhắc `source="raw"` + k-core tự đặt, hoặc `0core`, cho số báo cáo chính.

## Checklist M3

- [x] `positive_threshold` (mặc định 4.0, cho phép None).
- [x] Split 3 phần train/val/test tolerant (val chỉ cho user ≥3); **giữ `MIN_INTERACTIONS=2`**.
- [x] Train giữ `rating`/`timestamp` cho M4.
- [x] `GraphRecDataset`: `validDict`.
