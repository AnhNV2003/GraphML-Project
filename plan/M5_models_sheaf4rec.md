# M5 — Models: Popularity baseline + Sheaf4Rec thật + expressiveness

Trọng tâm khoa học. Mọi model theo API `BPRModelBase` trong
[extra_models.py](../gnn_recommendation/extra_models.py) để tái dùng `Procedure`/`utils`.

## 5.1 Popularity baseline (Table 1 proposal — bắt buộc)

Không cá nhân hoá: mọi user nhận cùng top-K item phổ biến nhất. Thêm `PopularityRec` vào
`extra_models.py`, **không train**:

```python
class PopularityRec(BPRModelBase):
    def __init__(self, config, dataset):
        super().__init__()
        counts = np.zeros(dataset.m_items)
        for pos in dataset.allPos:
            counts[pos] += 1
        self.item_score = nn.Parameter(torch.tensor(counts, dtype=torch.float32),
                                        requires_grad=False)
    def getUsersRating(self, users):
        return self.item_score.unsqueeze(0).expand(len(users), -1)
```

Pipeline cần cờ `trainable=False` → bỏ vòng `train_bpr_model`, chỉ evaluate. Đây là "sàn" để
chứng minh các model khác thật sự cá nhân hoá.

## 5.2 Sheaf4Rec thật — file mới `gnn_recommendation/sheaf.py`

Tham chiếu: Sheaf4Rec (Purificato et al., TORS 2025) + Neural Sheaf Diffusion (Bodnar et al.,
NeurIPS 2022; official code `twitter-research/neural-sheaf-diffusion` cho Laplacian builder +
sheaf learner).

### Cấu trúc

Graph bipartite `G=(U∪I,E)`, stalk dim `d`:
- Node `v` mang stalk `x_v ∈ R^d` (0-cochain); embedding table `Ψ` `(n_nodes, d)`, init normal 0.1.
- Cạnh `e=(u,i)`: restriction maps `F_{u◁e}, F_{i◁e} ∈ R^{d×d}`.
- Co-boundary: `δ(x)_e = F_{i◁e} x_i − F_{u◁e} x_u` (eq. 5).
- Sheaf Laplacian: `L_F = δ^T δ`, `Δ_F = D^{-1/2} L_F D^{-1/2}` (eq. 6).
- Diffusion layer (eq. 7): `X ← X − σ( Δ_F (I⊗W1) X W2 )`, `W1∈R^{d×d}`, `W2∈R^{f×f}`
  (mặc định `f=1` → bỏ `W2`).

### Sheaf learner + expressiveness `(N,1)/(1,N)/(N,N)`

`F_{v◁e} = Φ([x_v ; x_w])` với `Φ` MLP nhỏ trả ma trận `d×d`. Cấu hình qua `restriction_type`:

| nhãn (proposal) | `restriction_type` | Dạng restriction map |
|---|---|---|
| **GCN-like `(N,1)`** | `scalar` | vô hướng × `I_d` (sheaf trivial → diffusion chuẩn) |
| trung gian | `diagonal` | ma trận đường chéo `d` tham số |
| **GAT-like `(1,N)`** | `attention` | vô hướng phụ thuộc cả 2 node (giống attention) |
| **full sheaf `(N,N)`** | `general` | ma trận `d×d` đầy đủ — main model |

> Ký hiệu `(N,1)/(1,N)/(N,N)` = độ phong phú của restriction map. Expose 3 nhãn
> `gcn_like / gat_like / full_sheaf`. **Đối chiếu công thức chính xác trong paper Sheaf4Rec khi
> code** (mục verify M5).

> **⚠ Verify finding (đo khi kiểm M5) — cần sửa trước E3 (M7):** implementation hiện tại cho
> `gcn_like` và `gat_like` **cùng cấu trúc** (đều out_dim=1 → scalar × I_d, 81 params như nhau),
> chỉ khác activation (softplus vs sigmoid) → thực chất chỉ có **2 nấc** (scalar vs full), không
> phải 3 như proposal cần. Để E3 có 3 nấc thật:
> - **`gcn_like` (N,1) = restriction CỐ ĐỊNH identity, KHÔNG học** (bỏ qua `RestrictionLearner`)
>   → Δ_F rút về graph Laplacian chuẩn = "không có cấu trúc sheaf" (đúng nghĩa GCN reduction).
> - **`gat_like` (1,N) = scalar học per-edge** (attention hiện tại) — giữ.
> - **`full_sheaf` (N,N) = ma trận d×d học** — giữ.
> Không sửa thì E3 vẫn chạy nhưng chỉ chứng minh được "scalar < full", **không** tách được
> GCN-reduction vs GAT-reduction như proposal tuyên bố.

### Triển khai sparse

Xây `Δ_F` dạng block sparse `(n_nodes·d, n_nodes·d)`: mỗi cạnh đóng góp block `d×d`
`F_u^T F_u`, `F_i^T F_i` (chéo), `−F_u^T F_i`, `−F_i^T F_u` (off-diag); chuẩn hoá `D^{-1/2}`
theo block-degree; nhân `Δ_F @ X` bằng `torch.sparse`. Bộ nhớ ~`d²·|E|` → bắt đầu `d∈{2,3,4}`.
Graph bipartite thuần không tương thích trực tiếp → dùng projection giữ tính sheaf, thực hành
xây trên graph đối xứng hoá của M4.

### API model

```python
class Sheaf4Rec(BPRModelBase):
    def forward(self):
        X = self.embedding.weight.view(n_nodes, d)
        for _ in range(self.n_layers):
            L = self._build_sheaf_laplacian(X)     # sheaf learner phụ thuộc X
            X = X - sigma(L @ (X @ self.W1))        # eq.7 rút gọn (f=1)
        users, items = split(pool(X))               # (n_users,d), (n_items,d)
        return users, items                         # score = users @ items.T (eq.3)
```

`getUsersRating`, `bpr_loss` kế thừa `BPRModelBase`.

### Config (`config.py` / `make_config`)

```python
"sheaf_stalk_dim": 4, "sheaf_n_layers": 3,
"sheaf_restriction_type": "general",   # gcn_like | gat_like | full_sheaf
"sheaf_dropout": 0.0,
```

## 5.3 Dọn `extra_models.py`

- Giữ `NGCF` ("standard GNN" trong proposal), `GAT` (optional, cần torch-geometric).
- Bỏ/đổi tên `SheafDiffusionRec` cũ → thay bằng `Sheaf4Rec(restriction_type="scalar")`.
- `UltraGCN`, `PureMF`: phụ, để sau cờ `--extra-models`; **không** thay Popularity.

## 5.4 Model set cho bảng chính

| Nhóm | Model | Bắt buộc |
|---|---|---|
| Non-personalized | Popularity | ✅ |
| Graph recommender | LightGCN | ✅ |
| Standard GNN | NGCF (+GAT nếu có PyG) | ✅ |
| Sheaf (main) | Sheaf4Rec `full_sheaf` | ✅ |
| Sheaf ablation | Sheaf4Rec `gcn_like`, `gat_like` | ✅ (M7 expressiveness) |
| MF (phụ) | PureMF | tuỳ |

## Checklist M5

- [x] `PopularityRec` + nhánh no-train trong pipeline.
- [x] `sheaf.py`: stalk embedding, sheaf learner (MLP→restriction), sparse sheaf Laplacian, eq.7.
- [x] 3 biến thể `gcn_like / gat_like / full_sheaf`.
- [x] Config keys stalk_dim / n_layers / restriction_type.
- [ ] **Verify**: đối chiếu Sheaf4Rec + NSD; overfit 1 batch để chắc gradient chảy; Sheaf ≥ Popularity.
  - Done: toy full_sheaf overfit loss `0.689611 -> 0.001921`.
  - Deferred to M7: Sheaf ≥ Popularity needs enough epochs/hyperparameter runs, not a 1-epoch smoke.
- [x] Gỡ/đổi tên `SheafDiffusionRec` cũ; cập nhật `build_extra_models`.
- [x] **(xong, trước E3)** `gcn_like` → `identity` type: 0 learnable params, fixed I_d.
  3 nấc thật sự: gcn_like=0, gat_like=81, full_sheaf=216 params (với stalk_dim=4).
  Đồng thời: thay `_build_sheaf_laplacian + torch.sparse.mm` bằng `_sheaf_diffuse` dùng
  scatter-add trực tiếp → tránh dense gradient O(n_nodes²·d²) = ~26 GB/layer/pass.
