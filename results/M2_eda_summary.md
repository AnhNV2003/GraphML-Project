# M2 EDA Summary

## Dataset Statistics

| dataset | n_users | n_items | n_interactions | sparsity | rating_min | rating_max | rating_mean | positive_rating_ge_4 | positive_rating_ge_4_ratio | user_id_history_min | user_id_history_mean | user_id_history_median | user_id_history_p90 | user_id_history_p95 | user_id_history_p99 | user_id_history_max | item_id_history_min | item_id_history_mean | item_id_history_median | item_id_history_p90 | item_id_history_p95 | item_id_history_p99 | item_id_history_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| amazon_beauty_raw | 631986 | 112565 | 701528 | 0.999990 | 1.000000 | 5.000000 | 3.960245 | 500107 | 0.712882 | 1 | 1.110037 | 1.000000 | 1.000000 | 2.000000 | 3.000000 | 165 | 1 | 6.232204 | 2.000000 | 11.000000 | 21.000000 | 73.000000 | 1962 |
| movielens_1m | 6040 | 3706 | 1000209 | 0.955316 | 1.000000 | 5.000000 | 3.581564 | 575281 | 0.575161 | 20 | 165.597517 | 96.000000 | 400.000000 | 556.000000 | 906.660000 | 2314 | 1 | 269.889099 | 123.500000 | 729.500000 | 1051.500000 | 1784.900000 | 3428 |
| amazon_beauty_benchmark_0core_timestamp | 631986 | 112565 | 693929 | 0.999990 | 1.000000 | 5.000000 | 3.960676 | 494769 | 0.712997 | 1 | 1.098013 | 1.000000 | 1.000000 | 2.000000 | 3.000000 | 164 | 1 | 6.164696 | 2.000000 | 11.000000 | 21.000000 | 72.000000 | 1952 |
| amazon_beauty_benchmark_5core_timestamp | 253 | 356 | 2535 | 0.971855 | 1.000000 | 5.000000 | 4.314793 | 2122 | 0.837081 | 5 | 10.019763 | 7.000000 | 18.000000 | 24.000000 | 34.960000 | 63 | 5 | 7.120787 | 7.000000 | 10.000000 | 11.250000 | 13.000000 | 15 |

## Before/After K-Core

| dataset | stage | min_interactions | n_users | n_items | n_interactions | sparsity |
| --- | --- | --- | --- | --- | --- | --- |
| amazon_beauty_raw | before_kcore | 2 | 631986 | 112565 | 701528 | 0.999990 |
| amazon_beauty_raw | after_kcore | 2 | 23137 | 12625 | 58148 | 0.999801 |
| movielens_1m | before_kcore | 2 | 6040 | 3706 | 1000209 | 0.955316 |
| movielens_1m | after_kcore | 2 | 6040 | 3592 | 1000095 | 0.953903 |

## Data Configuration Decision

Amazon All Beauty raw is the preferred source for the main modelling pipeline after M3 filtering because it preserves the full local dataset before rating-threshold and k-core choices. The bundled 5-core benchmark is useful for smoke tests and reference-aligned experiments, but it is very small compared with raw/0-core data. MovieLens uses the local ratings.dat file as the canonical interaction source.
