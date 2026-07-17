# M4 — Graph Construction (thay cho "Feature Engineering")

Trong CF + GNN/sheaf không có feature thủ công; "kỹ thuật đặc trưng" thực chất là **cách dựng
graph và trọng số cạnh**. Đây cũng là đóng góp riêng mà proposal muốn thử (rating/time-aware).

Code trong `gnn_recommendation/graph.py`.

## 4.1 `build_normalized_graph` — graph binary (baseline)

Dựng adjacency bipartite đối xứng hóa, chuẩn hóa `D^{-1/2} A D^{-1/2}`, trả về
`torch.sparse_coo_tensor`. Đây là giao thức chuẩn hóa **dùng chung cho LightGCN, Sheaf4Rec
(cả bản tự viết `sheaf.py` và bản official `sheaf_official.py` tự xây Laplacian riêng từ
`allPos`), NCL, DirectAU** — chỉ NGCF dùng chuẩn hóa khác (`D^{-1}(A+I)`, xem M5).

## 4.2 Edge weighting có tham số hóa (`edge_mode`)

```python
def compute_edge_weight(train_df, edge_mode="binary", rating_scale=5.0):
    if edge_mode == "binary":
        return None
    if edge_mode == "rating":
        return np.clip(train_df["rating"].astype(float).to_numpy() / rating_scale, 1e-8, None)
    if edge_mode == "time":
        recency = (timestamps - t_min) / max(t_max - t_min, 1.0)
        return np.exp(-(1.0 - recency))
```

- **binary**: `edge_weight=None` → mọi cạnh trọng số 1 (mặc định).
- **rating**: `rating/5.0`, clip dương. Nên đặt `positive_threshold=None` ở M3 khi thử
  edge_mode này, để rating còn phân biệt được (nếu lọc `rating>=4` trước, mọi cạnh còn lại
  chỉ còn giá trị 4 hoặc 5 → tín hiệu yếu).
- **time**: `exp(-(1 - recency))`, recency chuẩn hóa `[0,1]` theo khoảng thời gian của
  chính train set. Timestamp Amazon là **ms** — không trộn với timestamp giây của
  MovieLens khi tính `t_min`/`t_max`.

`build_normalized_graph(..., edge_weight=edge_weight, edge_mode=edge_mode)` nhân đôi
weight cho cả 2 chiều `(u→i)`, `(i→u)` trước khi chuẩn hóa đối xứng, giữ đúng tính đối
xứng của graph.

## 4.3 Kết nối dữ liệu

M3 trả về DataFrame train giữ nguyên `rating`/`timestamp` → `pipeline.prepare_dataset()`
tính `edge_weight` tương ứng `edge_mode` rồi truyền vào `build_normalized_graph`. Graph này
là input chung cho mọi model (kể cả Sheaf4Rec bản tự viết `sheaf.py`, dùng `dataset.Graph`
để xây sheaf Laplacian).

`experiments/exp_edges.py` (E4, xem M7) sweep `edge_mode ∈ {binary, rating, time}` trên các
model trong `--models`.

## Checklist M4

- [x] `edge_mode` = binary | rating | time trong `build_normalized_graph`.
- [x] `compute_edge_weight` từ train DataFrame (rating & time).
- [x] Symmetric hóa weight đúng cho 2 chiều.
- [x] E4 (`exp_edges.py`) hỗ trợ `--models` để chọn model sweep.
