# M5 — Models: 5 model chuẩn + Sheaf4Rec expressiveness study + model phụ

Trọng tâm khoa học. Mọi model theo API `BPRModelBase`
([model_base.py](../gnn_recommendation/model_base.py)): `forward() -> (user_emb, item_emb)`,
`bpr_loss(users, pos, neg)`, `getUsersRating(users)` — để tái dùng `Procedure`/`utils` của
LightGCN-PyTorch (train loop, BPR loss, sampling) cho **mọi** model không phân biệt nguồn gốc.

## 5.1 Bảng chính: 5 model đồ thị chuẩn

| Model | Registry key | File | Paper |
|---|---|---|---|
| LightGCN | `LightGCN` | import trực tiếp từ `assets/external_repos/LightGCN-PyTorch` (qua `official.py`) | He et al., SIGIR'20 |
| NGCF | `NGCF` | [extra_models.py](../gnn_recommendation/extra_models.py) (`NGCFRec`) | Wang et al., SIGIR'19 |
| Sheaf4Rec | `Sheaf4Rec-official` | [sheaf_official.py](../gnn_recommendation/sheaf_official.py) | Purificato et al., ACM TORS'23/25 |
| NCL | `NCL` | [ssl_models.py](../gnn_recommendation/ssl_models.py) (`NCLRec`) | Lin et al., WWW'22 |
| DirectAU | `DirectAU` | [ssl_models.py](../gnn_recommendation/ssl_models.py) (`DirectAURec`) | Wang et al., KDD'22 |

`experiments/common.py:FIVE_MODELS` giữ danh sách này làm mặc định của `--models` cho mọi
script E0-E4. Mỗi model đã được **verify đối chiếu repo gốc** trong
`assets/external_repos/` (vendor: `LightGCN-PyTorch`, `NGCF-PyTorch-official`,
`Sheaf4Rec-official`, `SELFRec-official` cho NCL/DirectAU) trước khi coi là kết quả chính
thức — xem chi tiết kiến trúc từng model bên dưới.

### LightGCN

Import **trực tiếp** từ repo gốc `gusye1234/LightGCN-PyTorch` (không port lại) — đây là
model **duy nhất** trong dự án chạy code gốc y hệt tại runtime, qua
`gnn_recommendation/official.py:load_official_modules()`. Propagation: `D^{-1/2} A D^{-1/2}`,
mean-pool qua các lớp.

### NGCF (`extra_models.py:NGCFRec`)

Port trung thành từ `huangtinglin/NGCF-PyTorch` (reference PyTorch de facto cho NGCF gốc
TensorFlow). Khác LightGCN ở 3 điểm:

- Chuẩn hóa **`D^{-1}(A+I)`** (mean-normalized, self-loop) — không phải `D^{-1/2} A D^{-1/2}`.
- Mỗi lớp có **`W_gc`/`W_bi` học riêng** + `leaky_relu` + L2-normalize, không chỉ nhân
  adjacency thuần túy như LightGCN.
- Đọc ra bằng **concat** qua các lớp (`dim = latent_dim * (n_layers + 1)`), không mean-pool.

### Sheaf4Rec (`sheaf_official.py:Sheaf4RecOfficial`)

Port trung thành từ `antoniopurificato/Sheaf4Rec` (`models.py: SheafConvLayer + RecSysGNN`).
Đây là bản dùng cho **bảng so sánh chính** (registry key `Sheaf4Rec-official`), khác với bản
tự viết `sheaf.py` (dùng riêng cho E3 expressiveness study, xem 5.3).

Kiến trúc (đúng công thức gốc, không đơn giản hóa):

- **Restriction map vô hướng** (scalar, không phải ma trận `d×d`): `sheaf_learner =
  Linear(2*latent_dim, 1, bias=False)` rồi `tanh`.
- **Sheaf Laplacian `n×n`** (không phải block `n·d × n·d`) — vì restriction map là scalar.
- Chuẩn hóa đối xứng `(deg + 1)^{-1/2}` với **self-loop ngầm** (`+1` trong công thức).
- Mỗi lớp dùng chung 1 `linear` map; diffusion: `x ← x − step_size · L @ linear(x)`.
- Đọc ra bằng **concat** qua các lớp (giống công thức gốc `RecSysGNN`).

### NCL (`ssl_models.py:NCLRec`)

Port từ `RUCAIBox/NCL` (qua framework `Coder-Yu/SELFRec`, vendor tại
`assets/external_repos/SELFRec-official/model/graph/NCL.py`). Dùng chung encoder
LightGCN-style (`_LightGCNBase`), cộng thêm 2 loss self-supervised:

- **Structural neighbor SSL**: so khớp embedding node với embedding của chính nó sau
  `hyper_layers` bước lan truyền (structure-aware contrastive).
- **ProtoNCE**: cluster user/item embedding bằng `sklearn.cluster.MiniBatchKMeans` (thay
  `faiss.Kmeans` gốc, không có GPU faiss trong môi trường này), kéo embedding gần centroid
  cùng cluster.

> **Hành vi cần lưu ý khi đọc log ngắn (`--quick`, 1-2 epoch)**: `_maybe_e_step()` chỉ bắt
> đầu chạy k-means sau `warmup_epochs=20` (mặc định). Nếu tổng số epoch train ít hơn
> warmup, `_user_centroids` luôn `None` và **ProtoNCE loss không bao giờ kích hoạt** — NCL
> lúc đó chỉ còn structural SSL + BPR, không phản ánh đúng hành vi đầy đủ của model. Không
> phải bug, nhưng cần epoch đủ dài (như trong `train.sh`, `EPOCHS_MAIN=100`) để NCL chạy
> đúng thiết kế.

### DirectAU (`ssl_models.py:DirectAURec`)

Port từ `THUwangcy/DirectAU` (qua SELFRec). Đây là model **duy nhất trong 5 model không
dùng BPR/negative sampling** — thay bằng 2 thành phần loss trực tiếp trên embedding dương:

- **Alignment**: khoảng cách L2 bình phương giữa user và item embedding tương ứng (càng
  gần nhau càng tốt).
- **Uniformity**: đẩy embedding phân bố đều trên siêu cầu (tránh collapse).

```python
def bpr_loss(self, users, pos, neg):   # `neg` không dùng — không cần sampling âm
    align = self._alignment(user_emb[users], item_emb[pos])
    uniform = self.gamma * (self._uniformity(user_emb[users]) + self._uniformity(item_emb[pos])) / 2
    return align + uniform, ...
```

## 5.2 Model phụ (không nằm trong `FIVE_MODELS`, chạy qua stage riêng của `train.sh`)

| Model | Registry key | Stage `train.sh` | Ghi chú |
|---|---|---|---|
| Popularity | `Popularity` | không có stage riêng — cần gọi thủ công `--models Popularity,...` | Non-personalized, `trainable=False`, không train |
| PureMF | `PureMF` | `bash train.sh puremf` | Import trực tiếp từ `LightGCN-PyTorch` gốc, không port lại |
| MF + TAG-CF | `MF`, `MF+TAG-CF` | `bash train.sh tagcf` | Test-time message passing (xem M7) |
| SGL, SimGCL, LightGCL | `SGL`, `SimGCL`, `LightGCL` | `bash train.sh sota` (cùng NCL/DirectAU) | Cùng họ self-supervised, dùng chung encoder `_LightGCNBase` |
| GAT | `GAT` | tùy chọn `include_gat` | Cần `torch-geometric`; optional qua `--no-gat` |
| UltraGCN-stub | `UltraGCN-stub` | `include_auxiliary=True` | **KHÔNG phải UltraGCN thật** — xem cảnh báo dưới |

> **Cảnh báo `UltraGCN-stub`** (`extra_models.py:MFStubRec`): đây chỉ là matrix
> factorization thuần (forward trả thẳng embedding, không có graph propagation, không có
> loss đặc trưng của UltraGCN như beta-score negative weighting hay item-item constraint).
> Registry key cố tình đặt `UltraGCN-stub` (không phải `UltraGCN`) để không ai nhầm số liệu
> này với UltraGCN thật khi đọc bảng kết quả.

## 5.3 Sheaf4Rec expressiveness study — bản tự viết (`sheaf.py`)

Riêng cho **E3** (xem M7), *không* dùng cho bảng so sánh chính (bảng chính dùng
`Sheaf4Rec-official`). Mục tiêu: so sánh 3 mức độ phức tạp của restriction map trên cùng
1 kiến trúc diffusion tự viết.

| Registry key | `restriction_type` | Số tham số (stalk_dim=4) | Ý nghĩa |
|---|---|---|---|
| `Sheaf4Rec-gcn_like` | `identity` (alias `gcn_like`) | **0** — cố định `I_d`, không học | Sheaf trivial → diffusion rút về graph Laplacian chuẩn (tương đương GCN) |
| `Sheaf4Rec-gat_like` | `attention` (alias `gat_like`) | **81** — vô hướng học theo từng cạnh | Giống cơ chế attention |
| `Sheaf4Rec-full_sheaf` | `general` (alias `full_sheaf`) | **216** — ma trận `d×d` đầy đủ | Sheaf đầy đủ, biểu đạt mạnh nhất |

```python
RESTRICTION_ALIASES = {
    "gcn_like": "identity",   # 0 params, fixed I_d
    "gat_like": "attention",  # scalar learned per edge
    "full_sheaf": "general",  # full d x d learned matrix
}
```

Sheaf Laplacian dựng dạng **block sparse** `(n_nodes·d, n_nodes·d)`: mỗi cạnh đóng góp
block `d×d` (`F_u^T F_u`, `F_i^T F_i` chéo; `−F_u^T F_i`, `−F_i^T F_u` off-diagonal), lan
truyền qua `_sheaf_diffuse` bằng scatter-add trực tiếp trên cạnh (không dựng dense
Laplacian) — tránh OOM (bản dựng dense từng gây ~26GB/layer/pass, đã fix bằng cách này).

## Checklist M5

- [x] LightGCN: import trực tiếp từ LightGCN-PyTorch gốc.
- [x] NGCF: port trung thành từ `huangtinglin/NGCF-PyTorch`, verify chuẩn hóa/depth/readout.
- [x] Sheaf4Rec-official: port trung thành từ `antoniopurificato/Sheaf4Rec`.
- [x] NCL, DirectAU: port từ SELFRec, verify công thức loss.
- [x] Model phụ: Popularity, PureMF, MF+TAG-CF, SGL/SimGCL/LightGCL, GAT, UltraGCN-stub.
- [x] `sheaf.py`: 3 biến thể `gcn_like`(0 params)/`gat_like`(81)/`full_sheaf`(216) cho E3.
- [x] Mọi model đăng ký qua `build_all_models`/`build_extra_models`, dùng chung
      `train_bpr_model`/`evaluate_official` (M6).
