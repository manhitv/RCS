#!/usr/bin/env bash
# Quick end-to-end smoke test of the RCS pipeline.
#
# Stage 1: sample N generations on a small slice of GPQA.
# Stage 2: rank every Best-of-N metric on the cached generations.
#
# Expected runtime: ~5 min on one mid-range GPU.

set -euo pipefail

MODEL="${MODEL:-qwen2.5-3b}"
DATASET="${DATASET:-gpqa}"
N_SAMPLES="${N_SAMPLES:-10}"
FRACTION="${FRACTION:-0.1}"
SEED="${SEED:-42}"

echo "[1/2] Generation: model=$MODEL dataset=$DATASET N=$N_SAMPLES F=$FRACTION seed=$SEED"
python -m src.generation \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --n_samples "$N_SAMPLES" \
    --fraction_of_data_to_use "$FRACTION" \
    --seed "$SEED"

echo "[2/2] Ranking: every Best-of-N metric in one pass"
python -m src.ranking \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --n_samples "$N_SAMPLES" \
    --fraction_of_data_to_use "$FRACTION" \
    --self_certainty \
    --modex \
    --include_oracle \
    --seed "$SEED"

echo "Done. See results/ranking_logs.tsv for the metric table."
