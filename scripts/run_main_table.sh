#!/usr/bin/env bash
# Table 1 (main results): 5 models x 5 datasets x 3 seeds at N=10.
#
#   Models   : qwen2.5-3b, qwen2.5-7b, llama3.2-3b, llama3.1-8b, gemma2-9b
#   Datasets : sciq, gpqa, arith_long, gsm8k, formal_logic
#   Seeds    : 42, 44, 46
#
# Stage 1 generates samples once per (model, dataset, seed) and caches them;
# stage 2 computes every Best-of-N metric in a single pass.

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=("qwen2.5-3b" "qwen2.5-7b" "llama3.2-3b" "llama3.1-8b" "gemma2-9b")
DATASETS=("sciq" "gpqa" "arith_long" "gsm8k" "formal_logic")
SEEDS=(42 44 46)
N=10

declare -A FRACTION=(
    [sciq]=1
    [gpqa]=1
    [arith_long]=0.2
    [gsm8k]=0.15
    [formal_logic]=1
)
declare -A MAX_NEW_TOKENS=(
    [sciq]=32
    [gpqa]=32
    [arith_long]=512
    [gsm8k]=512
    [formal_logic]=512
)

for seed in "${SEEDS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    frac=${FRACTION[$dataset]}
    mnt=${MAX_NEW_TOKENS[$dataset]}
    for model in "${MODELS[@]}"; do
      echo "=== gen+rank | model=$model dataset=$dataset seed=$seed N=$N frac=$frac ==="

      python -m src.generation \
        --model "$model" --dataset "$dataset" \
        --n_samples "$N" --fraction_of_data_to_use "$frac" \
        --max_new_tokens "$mnt" --seed "$seed"

      python -m src.ranking \
        --model "$model" --dataset "$dataset" \
        --n_samples "$N" --fraction_of_data_to_use "$frac" \
        --self_certainty --modex --include_oracle --seed "$seed"
    done
  done
done

echo "=== Main table done — see results/ranking_logs.tsv ==="
