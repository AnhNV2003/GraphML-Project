# M2 EDA Summary

## Dataset Statistics

| dataset | n_users | n_items | n_interactions | sparsity | rating_min | rating_max | rating_mean | positive_rating_ge_4 | positive_rating_ge_4_ratio | user_id_history_min | user_id_history_mean | user_id_history_median | user_id_history_p90 | user_id_history_p95 | user_id_history_p99 | user_id_history_max | item_id_history_min | item_id_history_mean | item_id_history_median | item_id_history_p90 | item_id_history_p95 | item_id_history_p99 | item_id_history_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| amazon_video_games_raw | 2766656 | 137249 | 4624615 | 0.999988 | 1.000000 | 5.000000 | 4.047460 | 3445132 | 0.744955 | 1 | 1.671554 | 1.000000 | 3.000000 | 4.000000 | 9.000000 | 664 | 1 | 33.695072 | 4.000000 | 51.000000 | 118.000000 | 547.000000 | 18105 |
| movielens_1m | 6040 | 3706 | 1000209 | 0.955316 | 1.000000 | 5.000000 | 3.581564 | 575281 | 0.575161 | 20 | 165.597517 | 96.000000 | 400.000000 | 556.000000 | 906.660000 | 2314 | 1 | 269.889099 | 123.500000 | 729.500000 | 1051.500000 | 1784.900000 | 3428 |

## Before/After K-Core

| dataset | stage | min_interactions | n_users | n_items | n_interactions | sparsity |
| --- | --- | --- | --- | --- | --- | --- |
| amazon_video_games_raw | before_kcore | 5 | 2766656 | 137249 | 4624615 | 0.999988 |
| amazon_video_games_raw | after_kcore | 5 | 95007 | 25838 | 816455 | 0.999667 |
| movielens_1m | before_kcore | 5 | 6040 | 3706 | 1000209 | 0.955316 |
| movielens_1m | after_kcore | 5 | 6040 | 3416 | 999611 | 0.951552 |

## Data Configuration Decision

Amazon Video Games raw is the preferred source for the main modelling pipeline after M3 filtering because it preserves the full local dataset before rating-threshold and k-core choices. The bundled 5-core benchmark is useful for smoke tests and reference-aligned experiments, but it is very small compared with raw/0-core data. MovieLens uses the local ratings.dat file as the canonical interaction source.
