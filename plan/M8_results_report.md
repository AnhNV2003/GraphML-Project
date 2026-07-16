# M8 — Results & Report

Đóng gói kết quả thành bảng + hình cho báo cáo NeurIPS và cập nhật tài liệu.

## 8.1 Chạy full

- Chạy E0–E4 (M7) trên Amazon (chính) + MovieLens (phụ), ≥3 seed, epoch đầy đủ.
- Xác nhận mọi kết quả `is_real_data=True`. Lưu raw CSV + config vào `results/`.

## 8.2 Bảng cho báo cáo

- **Bảng chính** (từ `main_comparison.csv`): các model × metric, Recall@10/NDCG@10 in đậm,
  kèm train/infer time. Format mean±std.
- **Bảng expressiveness** (E3), **bảng edge construction** (E4).
- Xuất LaTeX table (pandas `to_latex`) để nhúng thẳng.

## 8.3 Hình (mở rộng `gnn_recommendation/plots.py`)

- `plot_layer_sweep` (E1), `plot_dim_tradeoff` (E2), `plot_expressiveness_bars` (E3),
  `plot_edge_bars` (E4), + EDA figures (M2).
- Style thống nhất, lưu PNG + PDF vào `results/figures/` cho LaTeX.

## 8.4 Viết mục Results/Analysis (theo cấu trúc proposal §5)

- **Outcome**: Sheaf4Rec vs Popularity/LightGCN/NGCF trên Recall@10, NDCG@10, F1, MRR;
  bàn về cân bằng precision–recall.
- **Mechanism**: over-smoothing (E1), accuracy–cost (E2), expressiveness (E3), edge construction (E4),
  và "slower to train, faster to serve" (timing M6).

## 8.5 Cập nhật docs

- `README.md`: cách chạy experiments, mô tả model mới, nguồn data local.
- `DATASET_COLUMNS.md`: đồng bộ schema thực tế.
- `requirements.txt`: pin version; thêm dependency Sheaf4Rec cần (nếu có).
- Ghi rõ khác biệt so với paper gốc (faithful re-implementation).

## Checklist M8

- [ ] Full run E0–E4, `is_real_data=True`, lưu raw + config.
- [ ] Bảng chính + expressiveness + edge (LaTeX).
- [ ] 4 hình phân tích + EDA figures.
- [ ] Viết Results/Analysis (outcome + mechanism).
- [ ] Cập nhật README / DATASET_COLUMNS / requirements.
