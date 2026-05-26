#!/usr/bin/env bash
# Quick end-to-end validation of the RCS pipeline.
# Generates 10 samples on a 10% slice of GPQA and ranks every metric.
# Expected runtime: a few minutes on one mid-range GPU.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen2.5-3b}"
DATASET="${DATASET:-gpqa}"
N="${N:-10}"
FRACTION="${FRACTION:-0.1}"
SEED="${SEED:-42}"

echo "=== [1/2] Generation (model=$MODEL dataset=$DATASET N=$N fraction=$FRACTION) ==="
python -m src.generation \
    --model "$MODEL" --dataset "$DATASET" \
    --n_samples "$N" --fraction_of_data_to_use "$FRACTION" --seed "$SEED"

echo "=== [2/2] Ranking (every metric in one pass) ==="
python -m src.ranking \
    --model "$MODEL" --dataset "$DATASET" \
    --n_samples "$N" --fraction_of_data_to_use "$FRACTION" \
    --self_certainty --modex --include_oracle --seed "$SEED"

echo "=== Validation complete — see results/ranking_logs.tsv ==="
