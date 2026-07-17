# Vì sao accuracy trên test thấp? — Phân tích cold-start & global timestamp split

> **k-core** = 5 · **Split**: global timestamp (như nhau cho mọi dataset)
> Số liệu tính trực tiếp từ pipeline (`prepare_dataset(<dataset>)`).

## TL;DR

Cùng một giao thức global timestamp split, nhưng **mức độ cold-start khác nhau rất lớn**
giữa 2 dataset, vì tốc độ tăng trưởng theo thời gian của chúng khác nhau:

| Dataset | Test pairs unhittable | Trần Recall lý thuyết | Mức độ ảnh hưởng |
|---|---:|---:|---|
| **Amazon Video Games** | **74.7%** | **~25.3%** | Nặng — metric tuyệt đối gần như vô nghĩa nếu đọc trên nền 100% |
| **MovieLens 1M** | **0.17%** | **~99.8%** | Không đáng kể — split gần như không gây cold-start |

Kết luận: cold-start **không phải thuộc tính của global timestamp split nói chung**, mà phụ
thuộc mạnh vào việc dataset có đang tăng trưởng nhanh theo thời gian hay không (nhiều
user/item mới xuất hiện gần cuối). Đây là lý do phải phân tích riêng từng dataset thay vì
dùng chung 1 con số.

---

## 1. Amazon Video Games

### 1.1 Chuỗi xử lý dữ liệu

| Bước | Số interactions | Ghi chú |
|---|---:|---|
| Raw (Video_Games.jsonl) | 4,624,615 | Toàn bộ review |
| Lọc positive `rating ≥ 4.0` | 3,445,132 | Giữ implicit positive feedback |
| **K-core `k = 5`** (lặp tới ổn định) | **536,363** | Mỗi user/item có ≥5 tương tác *tổng cộng* |
| Global timestamp split | 450,813 / 55,460 / 30,090 | train / valid / test |

Sau lọc: **63,797 users · 19,347 items · 536,363 interactions** (mật độ ~0.043%).

### 1.2 Tỷ lệ split (theo thời gian tuyệt đối)

Một cặp cutoff chung cho cả dataset: `t1 = 1588193961423`, `t2 = 1654879144964` (ms-epoch).

| Tập | Điều kiện | #pairs | % |
|---|---|---:|---:|
| Train | `t < t1` | 450,813 | 84.05% |
| Valid | `t1 ≤ t < t2` | 55,460 | 10.34% |
| Test | `t ≥ t2` | 30,090 | 5.61% |

Tỷ lệ khớp đúng target quantile (84.05 / 10.34 / 5.61) → split chạy đúng thiết kế.

### 1.3 Cold-start — gốc rễ của accuracy thấp

**Cold-start ở phía USER:**

| Chỉ số | Giá trị | % trên test user |
|---|---:|---:|
| Test user phân biệt | 8,774 | 100% |
| — có lịch sử trong train (**warm**, đánh giá có ý nghĩa) | 5,785 | **65.9%** |
| — KHÔNG có trong train (**cold**, model chưa từng thấy) | 2,989 | **34.1%** |
| Tương tác test thuộc về user cold | 17,132 | 56.9% |

→ Hơn 1/3 test user không có embedding học được. Với họ, model chỉ có thể đoán mù.

**Cold-start ở phía ITEM (nghiêm trọng hơn):**

| Chỉ số | Giá trị | % |
|---|---:|---:|
| Test item phân biệt | 5,908 | 100% |
| — chưa từng thấy trong train (không thể xếp hạng) | 2,949 | **49.9%** |
| **Tương tác test có item unhittable** | **22,482 / 30,090** | **74.7%** |

Item chưa có trong train thì không có embedding → không bao giờ lọt top-K → tương tác
đó **không thể recall được về mặt lý thuyết**.

**Trần accuracy lý thuyết:**

```
Recall ceiling = 1 − (test pairs có item cold) / (tổng test pairs)
               = 1 − 22,482 / 30,090
               ≈ 25.3%
```

Nghĩa là **dù model hoàn hảo tuyệt đối, Recall@∞ cũng không vượt quá ~25%**, và Recall@10
(chỉ 10 slot) còn thấp hơn nhiều. Con số thực đo được 0.5% cần đọc trên nền trần đã bị cắt
này, không phải trên nền 100%.

### 1.4 Vì sao k = 5 vẫn sinh ra cold-start?

K-core đảm bảo mỗi user/item có **≥5 tương tác trên toàn bộ dataset**, nhưng **không** ràng
buộc 5 tương tác đó nằm ở khoảng thời gian nào. Split lại cắt theo **một mốc thời gian chung**:

```
Trục thời gian:  ────t1──────────t2──────────────►
User A (5 review):                 ● ● ● ● ●      ← cả 5 đều sau t2
                 └── train ──┘└ valid ┘└─ test ─┘
                    (rỗng)      (rỗng)    (5 pairs)
```

User A qua k-core (5 ≥ 5 ✓) nhưng có **0 train / 0 valid / 5 test** → cold-start.

Video Games là category **tăng trưởng nhanh theo thời gian** (số review/tháng tăng mạnh
qua các năm) → phần lớn user/item mới đăng ký/ra mắt dồn về cuối trục thời gian → rơi hết
vào kỳ test.

### 1.5 Kết quả E0 dưới lăng kính này

| Rank | Model | test Recall@10 | test NDCG@10 | val NDCG@10 |
|---|---|---:|---:|---:|
| 🥇 | NCL | 0.0072 | 0.0048 | 0.0131 |
| 🥈 | LightGCN | 0.0054 | 0.0036 | 0.0105 |
| 🥉 | NGCF | 0.0045 | 0.0030 | 0.0086 |
| 4 | DirectAU | 0.0043 | 0.0030 | 0.0089 |
| 5 | Sheaf4Rec-official | 0.0039 | 0.0026 | 0.0076 |

Nhận xét:
- **val NDCG cao hơn test NDCG đồng đều ở mọi model** (~2.7×) — đúng như dự đoán: valid gần
  train hơn về thời gian nên ít cold-start hơn test. Đây là bằng chứng cross-check rằng
  chênh lệch đến từ split, không từ overfitting của riêng model nào.
- Thứ hạng nhất quán (NCL > LightGCN > NGCF ≈ DirectAU > Sheaf4Rec) — kết luận so sánh vẫn vững.

---

## 2. MovieLens 1M

### 2.1 Chuỗi xử lý dữ liệu

| Bước | Số interactions | Ghi chú |
|---|---:|---|
| Raw (ratings.dat) | 1,000,209 | Toàn bộ rating |
| Lọc positive `rating ≥ 4.0` | 575,281 | Giữ implicit positive feedback |
| **K-core `k = 5`** (lặp tới ổn định) | **574,376** | Gần như không đổi — dataset đã dày đặc sẵn |
| Global timestamp split | 482,762 / 59,391 / 32,223 | train / valid / test |

Sau lọc: **6,034 users · 3,125 items · 574,376 interactions** (mật độ ~3.05% — **dày đặc
hơn Video Games gấp ~70 lần**, vì K-core gần như không loại ai: 575,281 → 574,376, chỉ mất
0.16% interaction).

### 2.2 Tỷ lệ split (theo thời gian tuyệt đối)

Cutoff tự suy qua `quantile_timestamp_cutoffs()`: `t1 = 976258125`, `t2 = 987309233` (s-epoch).

| Tập | Điều kiện | #pairs | % |
|---|---|---:|---:|
| Train | `t < t1` | 482,762 | 84.05% |
| Valid | `t1 ≤ t < t2` | 59,391 | 10.34% |
| Test | `t ≥ t2` | 32,223 | 5.61% |

Cùng tỷ lệ mục tiêu 84.05/10.34/5.61 như Video Games — vì cùng dùng
`quantile_timestamp_cutoffs`, chỉ khác đơn vị thời gian (giây thay vì mili-giây).

### 2.3 Cold-start — gần như không tồn tại

**Cold-start ở phía USER:**

| Chỉ số | Giá trị | % trên test user |
|---|---:|---:|
| Test user phân biệt | 891 | 100% |
| — có lịch sử trong train (**warm**) | 792 | **88.9%** |
| — KHÔNG có trong train (**cold**) | 99 | **11.1%** |
| Tương tác test thuộc về user cold | 4,277 | 13.3% |

**Cold-start ở phía ITEM (gần như không có):**

| Chỉ số | Giá trị | % |
|---|---:|---:|
| Test item phân biệt | 2,700 | 100% |
| — chưa từng thấy trong train | 5 | **0.19%** |
| **Tương tác test có item unhittable** | **54 / 32,223** | **0.17%** |

**Trần accuracy lý thuyết:**

```
Recall ceiling = 1 − 54 / 32,223 ≈ 99.8%
```

Gần như toàn bộ tương tác test đều **có thể recall được về lý thuyết** — khác hoàn toàn so
với Video Games (trần chỉ ~25.3%). Nếu 5 model chạy trên MovieLens, metric tuyệt đối đo
được sẽ phản ánh gần đúng năng lực thật của model, không bị "cắt trần" bởi cold-start.

### 2.4 Vì sao MovieLens gần như không có cold-start?

MovieLens 1M là dataset **tĩnh, đã đóng băng theo thời gian** (thu thập trong ~3 năm,
2000–2003, với số lượng user/phim cố định ngay từ đầu — không có "user mới đăng ký" hay
"phim mới ra mắt" liên tục như một nền tảng thương mại điện tử đang phát triển). Vì vậy:

- Hầu hết user đã rated phim xuyên suốt cả giai đoạn thu thập → dù cắt ở đâu trên trục thời
  gian, đa số user vẫn có interaction ở cả 3 phía train/valid/test.
- Danh sách phim gần như cố định ngay từ đầu (không có phim mới ra rạp liên tục trong tập dữ
  liệu) → item hầu như luôn xuất hiện trong train trước khi xuất hiện lại ở test.

→ Global timestamp split trên MovieLens hoạt động gần giống một split ngẫu nhiên theo tỷ lệ,
**không** tạo áp lực cold-start đáng kể như trên Video Games.

### 2.5 Trạng thái training hiện tại

**Chưa có kết quả 5 model chuẩn trên MovieLens 1M.** `results/main_comparison.csv` (E0)
hiện chỉ chứa `amazon_video_games` — `train.sh e0` chưa được chạy với
`DATASET_MAIN=movielens_1m` (hoặc target `all` chưa bao gồm MovieLens trong lần chạy gần
nhất). Vì trần Recall lý thuyết trên MovieLens gần 100%, đây sẽ là phép đối chứng tốt để
xác nhận: nếu metric tuyệt đối trên MovieLens **cao hơn hẳn** Video Games (như kỳ vọng vì
không bị cold-start bóp nghẹt), điều đó càng củng cố luận điểm ở Mục 1 rằng con số thấp trên
Video Games đến từ đặc tính split, không phải model yếu.

---

## 3. So sánh chéo & bài học

| | Amazon Video Games | MovieLens 1M |
|---|---:|---:|
| Users sau lọc | 63,797 | 6,034 |
| Items sau lọc | 19,347 | 3,125 |
| Mật độ (density) | ~0.043% | ~3.05% |
| Cold-start user (test) | 34.1% | 11.1% |
| Cold-start item (test) | 49.9% | 0.19% |
| Test pairs unhittable | 74.7% | 0.17% |
| Trần Recall lý thuyết | ~25.3% | ~99.8% |
| Bản chất dataset | Nền tảng thương mại đang tăng trưởng, liên tục có user/sản phẩm mới | Snapshot tĩnh, khép kín theo thời gian |

**Bài học chính:** Global timestamp split là giao thức đánh giá trung thực với thực tế
production, nhưng mức độ "khắc nghiệt" của nó phụ thuộc hoàn toàn vào tính chất tăng trưởng
của dataset — không thể so sánh trực tiếp Recall@10 tuyệt đối giữa 2 dataset dùng cùng 1
split protocol nếu không kèm theo phân tích cold-start này. Khi viết report, nên luôn đính
kèm bảng Mục 3 để hội đồng chấm hiểu vì sao cùng một model có thể có metric tuyệt đối chênh
lệch rất lớn giữa 2 dataset, dù thứ hạng tương đối giữa các model vẫn là điều có ý nghĩa
khoa học chính.

## 4. Nếu muốn giảm cold-start trên Video Games / gần literature hơn

Ba lựa chọn (chưa áp dụng — mặc định giữ nguyên global timestamp để trung thực với benchmark):

1. **Evaluate chỉ trên warm test user** (bỏ cold user khi tính metric) — phổ biến nhất
   trong literature, không đổi split, chỉ đổi tập user đánh giá.
2. **Áp k-core sau split** (đảm bảo mỗi user có ≥1 train) — thay đổi tập dữ liệu.
3. **Per-user temporal split** (mỗi user cắt theo % thời gian riêng) — thỏa hiệp giữa
   leave-one-out và global timestamp.

---

_File này giải thích tại sao accuracy tuyệt đối thấp trên test, so sánh giữa 2 dataset._
_Số liệu nguồn: `results/main_comparison.csv`, `results/logs/video_games_full.log`, và tính_
_trực tiếp từ `prepare_dataset('amazon_video_games')` / `prepare_dataset('movielens_1m')`._
