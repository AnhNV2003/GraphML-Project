# M2 — Exploratory Data Analysis

Dùng `gnn_recommendation/eda.py` chạy trên **real data local** để lấy số cho phần "Dataset"
của báo cáo và để biện minh các lựa chọn tiền xử lý ở M3.

## 2.1 Thống kê cần lấy (cho cả Amazon All Beauty và MovieLens 1M)

- `#users`, `#items`, `#interactions`, **sparsity** = 1 − |E|/(|U|·|I|).
- Phân bố **độ dài lịch sử** mỗi user và mỗi item (long-tail) → biện minh k-core.
- Phân bố **rating** (1–5) → biện minh ngưỡng implicit `rating ≥ 4` ở M3 (cho thấy tỉ lệ
  positive còn lại là hợp lý, không quá thưa).
- Phân bố **timestamp** theo thời gian → biện minh time-aware split / time-aware edges (M4).
- Trước/sau k-core: bảng so sánh #users/#items/#interactions để thấy tác động lọc.

## 2.2 Output

- Số liệu vào bảng "Dataset statistics" của báo cáo.
- Hình vào `results/figures/` (histogram độ dài lịch sử, phân bố rating, số tương tác theo thời gian).
- Cập nhật `DATASET_COLUMNS.md` nếu schema thực tế khác mô tả.

## 2.3 Lưu ý quan trọng

- Chạy EDA **trước khi** áp ngưỡng rating và k-core (để thấy dữ liệu gốc), rồi chạy lại
  **sau** tiền xử lý (để thấy dữ liệu đưa vào model).
- All Beauty 5-core rất nhỏ (~2k tương tác train). EDA giúp quyết định: dùng 0core hay raw
  + k-core tự đặt. Ghi lại lựa chọn và lý do.

## Checklist M2

- [x] Chạy `eda.py` trên Amazon (raw) + MovieLens, lưu số liệu + hình.
- [x] Bảng thống kê dataset (trước/sau tiền xử lý).
- [x] Chốt cấu hình dữ liệu (core level) dựa trên EDA.
