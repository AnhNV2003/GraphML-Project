#!/usr/bin/env bash
# Full training script — 5 canonical graph models on Amazon Video Games + MovieLens 1M.
#
# Models  : LightGCN (SIGIR'20), NGCF (SIGIR'19), Sheaf4Rec-official (ACM TORS'23/25),
#           NCL (WWW'22), DirectAU (KDD'22)  -- see plan/README.md for full citations.
# Split   : global timestamp split only (Amazon Reviews 2023 benchmark style).
#
# Usage:
#   bash train.sh                     # run everything (E0-E4 + SOTA + TAG-CF + PureMF)
#   bash train.sh e0                  # only E0 (main comparison)
#   bash train.sh e1                  # only E1 (layer sweep)
#   bash train.sh e2                  # only E2 (dim sweep)
#   bash train.sh e3                  # only E3 (Sheaf4Rec restriction-map study)
#   bash train.sh e4                  # only E4 (edge construction)
#   bash train.sh sota                # only the 5 SOTA self-supervised models
#   bash train.sh tagcf               # only MF + TAG-CF
#   bash train.sh puremf              # only PureMF baseline
#   bash train.sh popularity          # only Popularity baseline (non-personalized)
#
#   MODELS="LightGCN,NGCF" bash train.sh e0     # restrict E0 to a subset of models
#   DATASET_MAIN=amazon_video_games bash train.sh e0 # restrict to one dataset
#
# GPU: automatically used when CUDA is available (configured in gnn_recommendation/config.py)
# Logs: written to results/logs/

set -euo pipefail

PYTHON="${PYTHON:-venv/bin/python}"
TARGET="${1:-all}"

# ── Dataset & reproducibility ────────────────────────────────────────────────
DATASET_MAIN="${DATASET_MAIN:-amazon_video_games,movielens_1m}"   # E0, SOTA, TAG-CF, PureMF: both datasets
DATASET_SWEEP="${DATASET_SWEEP:-amazon_video_games}"              # E1-E4: Amazon only (faster iteration)
SEEDS="${SEEDS:-42,43,44}"

# ── 5 canonical graph models (override with MODELS="A,B" to restrict) ───────
MODELS="${MODELS:-LightGCN,NGCF,Sheaf4Rec-official,NCL,DirectAU}"

# ── Epoch budgets ─────────────────────────────────────────────────────────────
EPOCHS_MAIN="${EPOCHS_MAIN:-100}"   # E0/SOTA/TAG-CF/PureMF: train to convergence
EPOCHS_SWEEP="${EPOCHS_SWEEP:-50}"  # E1-E4: ablation sweeps
PATIENCE="${PATIENCE:-10}"          # early stopping patience (epochs without val improvement)

# ── Output ───────────────────────────────────────────────────────────────────
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR" results

run() {
    local label="$1"; shift
    echo ""
    echo "================================================================"
    echo "  $label"
    echo "================================================================"
    "$PYTHON" "$@" 2>&1 | tee "$LOG_DIR/${label}.log"
    echo "[done] $label -> $LOG_DIR/${label}.log"
}

# ── E0: Main comparison table (5 canonical models) ───────────────────────────
run_e0() {
    run "e0_main" -m experiments.exp_main \
        --dataset "$DATASET_MAIN" \
        --seeds   "$SEEDS" \
        --epochs  "$EPOCHS_MAIN" \
        --patience "$PATIENCE" \
        --no-gat \
        --models "$MODELS" \
        --output  results/main_comparison.csv
}

# ── E1: Layer sweep ───────────────────────────────────────────────────────────
run_e1() {
    run "e1_layers" -m experiments.exp_layers \
        --dataset "$DATASET_SWEEP" \
        --seeds   "$SEEDS" \
        --epochs  "$EPOCHS_SWEEP" \
        --patience "$PATIENCE" \
        --layers  "1,2,3,4,5,6,8" \
        --models  "$MODELS" \
        --output  results/layer_sweep.csv
}

# ── E2: Embedding-dimension sweep ─────────────────────────────────────────────
run_e2() {
    run "e2_dim" -m experiments.exp_dim \
        --dataset      "$DATASET_SWEEP" \
        --seeds        "$SEEDS" \
        --epochs       "$EPOCHS_SWEEP" \
        --patience     "$PATIENCE" \
        --latent-dims  "16,32,64,128" \
        --sheaf-dims   "1,2,3,4,6,8" \
        --models       "$MODELS" \
        --output       results/dim_sweep.csv
}

# ── E3: Sheaf4Rec restriction-map expressiveness study ────────────────────────
# gcn_like=0 learned params (fixed identity), gat_like=81, full_sheaf=216
run_e3() {
    run "e3_expressiveness" -m experiments.exp_expressiveness \
        --dataset "$DATASET_SWEEP" \
        --seeds   "$SEEDS" \
        --epochs  "$EPOCHS_SWEEP" \
        --patience "$PATIENCE" \
        --output  results/expressiveness.csv
}

# ── E4: Edge construction ─────────────────────────────────────────────────────
run_e4() {
    run "e4_edges" -m experiments.exp_edges \
        --dataset    "$DATASET_SWEEP" \
        --seeds      "$SEEDS" \
        --epochs     "$EPOCHS_SWEEP" \
        --patience   "$PATIENCE" \
        --edge-modes "binary,rating,time" \
        --models     "$MODELS" \
        --output     results/edge_construction.csv
}

# ── SOTA: 5 self-supervised graph-CF models (SGL/SimGCL/DirectAU/NCL/LightGCL) ─
run_sota() {
    run "sota_full" -m experiments.exp_sota \
        --dataset "$DATASET_MAIN" \
        --seeds   "$SEEDS" \
        --epochs  "$EPOCHS_MAIN" \
        --patience "$PATIENCE" \
        --output  results/sota_comparison.csv
}

# ── TAG-CF: MF + test-time message-passing aggregation ────────────────────────
run_tagcf() {
    run "tagcf_full" -m experiments.exp_tagcf \
        --dataset "$DATASET_MAIN" \
        --seeds   "$SEEDS" \
        --epochs  "$EPOCHS_MAIN" \
        --patience "$PATIENCE" \
        --output  results/tagcf_comparison.csv
}

# ── PureMF: official LightGCN-PyTorch matrix-factorization baseline ──────────
run_puremf() {
    run "puremf_full" -m experiments.exp_puremf \
        --dataset "$DATASET_MAIN" \
        --seeds   "$SEEDS" \
        --epochs  "$EPOCHS_MAIN" \
        --patience "$PATIENCE" \
        --output  results/puremf_comparison.csv
}

# ── Popularity: non-personalized top-popular-item baseline (no training) ──────
run_popularity() {
    run "popularity_full" -m experiments.exp_popularity \
        --dataset "$DATASET_MAIN" \
        --seeds   "$SEEDS" \
        --epochs  "$EPOCHS_MAIN" \
        --patience "$PATIENCE" \
        --output  results/popularity_comparison.csv
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$TARGET" in
    e0) run_e0 ;;
    e1) run_e1 ;;
    e2) run_e2 ;;
    e3) run_e3 ;;
    e4) run_e4 ;;
    sota) run_sota ;;
    tagcf) run_tagcf ;;
    puremf) run_puremf ;;
    popularity) run_popularity ;;
    all)
        run_e0
        run_e1
        run_e2
        run_e3
        run_e4
        run_sota
        run_tagcf
        run_puremf
        run_popularity
        echo ""
        echo "================================================================"
        echo "  All training complete. Results in results/"
        echo "  Next: bash test.sh   (builds comparison tables + figures)"
        echo "================================================================"
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: bash train.sh [all|e0|e1|e2|e3|e4|sota|tagcf|puremf|popularity]"
        exit 1
        ;;
esac
