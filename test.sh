#!/usr/bin/env bash
# Build the final comparison tables + figures from results/*.csv produced by train.sh.
#
# Usage:
#   bash test.sh              # just build tables/figures from existing results/*.csv
#   bash test.sh --smoke      # first re-run every experiment in --quick mode (fast
#                              # correctness check, ~1 epoch each) before building the report
#
# Output:
#   results/M8_comparison.md               combined ranked table (all datasets)
#   results/tables/model_comparison_*.md    per-dataset ranked table
#   results/figures/model_comparison_*.png  per-dataset bar chart (Recall@10 & NDCG@10)
#   results/figures/e1_layer_sweep.png, e2_dim_sweep_*.png, e3_expressiveness.png,
#     e4_edge_construction.png              ablation-sweep line charts (only if the
#                                            corresponding results/*.csv exists)

set -euo pipefail

PYTHON="${PYTHON:-venv/bin/python}"
MODE="${1:-report}"

if [[ "$MODE" == "--smoke" ]]; then
    echo "=== Smoke-testing every experiment script (--quick, ~1 epoch each) ==="
    "$PYTHON" -m experiments.exp_main --quick --no-gat --output results/main_comparison.csv
    "$PYTHON" -m experiments.exp_layers --quick --output results/layer_sweep.csv
    "$PYTHON" -m experiments.exp_dim --quick --output results/dim_sweep.csv
    "$PYTHON" -m experiments.exp_expressiveness --quick --output results/expressiveness.csv
    "$PYTHON" -m experiments.exp_edges --quick --output results/edge_construction.csv
    "$PYTHON" -m experiments.exp_sota --quick --output results/sota_comparison.csv
    "$PYTHON" -m experiments.exp_tagcf --quick --output results/tagcf_comparison.csv
    "$PYTHON" -m experiments.exp_puremf --quick --output results/puremf_comparison.csv
    echo "=== Smoke test complete ==="
fi

echo ""
echo "=== Building comparison tables + figures from results/ ==="
"$PYTHON" -m experiments.build_report

echo ""
echo "================================================================"
echo "  Done. See:"
echo "    results/M8_comparison.md"
echo "    results/tables/model_comparison_*.md"
echo "    results/figures/model_comparison_*.png"
echo "================================================================"
