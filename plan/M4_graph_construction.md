# M4 — Graph Construction (thay cho "Feature Engineering")

Trong CF + GNN/sheaf không có feature thủ công; "kỹ thuật đặc trưng" thực chất là **cách dựng
graph và trọng số cạnh**. Đây cũng là "đóng góp riêng" mà proposal muốn thử (rating/time-aware).

Sửa `gnn_recommendation/graph.py`.

## 4.1 Đã có (giữ)

`build_normalized_graph`: dựng adjacency bipartite đối xứng hoá, chuẩn hoá `D^{-1/2} A D^{-1/2}`,
trả `torch.sparse_coo_tensor`. Đây là edge **binary** — baseline của so sánh.

## 4.2 Edge weighting có tham số hoá

```python
def build_normalized_graph(n_users, n_items, pairs, device,
                           edge_weight=None, edge_mode="binary"):
    # edge_mode: "binary" | "rating" | "time"
    ...
    values = edge_weight if edge_weight is not None else np.ones(len(row))
    # sau đó vẫn D^{-1/2} A D^{-1/2} như cũ, chỉ thay `values`
```

- **binary**: như hiện tại.
- **rating**: weight = rating chuẩn hoá, ví dụ `rating/5` hoặc `(rating-3)` (dương). Lấy từ
  `train["rating"]`. Khi thử rating-aware nên đặt `positive_threshold=None` ở M3 để rating còn
  mang thông tin.
- **time**: weight tăng theo độ mới, ví dụ `exp(-λ·(t_max - t))` (λ chuẩn hoá theo phạm vi
  timestamp) hoặc rank-recency trong mỗi user. Nhớ timestamp Amazon là **ms**.

Chuẩn hoá đối xứng giữ nguyên; symmetric hoá phải nhân đôi weight cho cả 2 chiều `(u→i)`, `(i→u)`.

## 4.3 Kết nối dữ liệu

M3 trả DataFrame train giữ `rating`/`timestamp` → `pipeline`/`experiments` tính `edge_weight`
tương ứng `edge_mode` rồi truyền vào `build_normalized_graph`. Với Sheaf4Rec (M5), graph này
là input để dựng sheaf Laplacian.

## Checklist M4

- [x] `edge_mode` = binary | rating | time trong `build_normalized_graph`.
- [x] Hàm tính `edge_weight` từ train DataFrame (rating & time).
- [x] Symmetric hoá weight đúng cho 2 chiều.
