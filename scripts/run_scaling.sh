#!/usr/bin/env bash
# Scaling experiment (Figure 1): how Best-of-N accuracy scales with N.
#
#   N ∈ {5, 10, 20, 40}  for two representative models on five datasets.
#
# Reuses cached generations when N matches an earlier run.

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=("qwen2.5-3b" "llama3.2-3b")
DATASETS=("sciq" "gpqa" "arith_long" "gsm8k" "formal_logic")
SEEDS=(42 44 46)
N_LIST=(5 10 20 40)

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
      for n in "${N_LIST[@]}"; do
        echo "=== scaling | model=$model dataset=$dataset seed=$seed N=$n frac=$frac ==="

        python -m src.generation \
          --model "$model" --dataset "$dataset" \
          --n_samples "$n" --fraction_of_data_to_use "$frac" \
          --max_new_tokens "$mnt" --seed "$seed"

        python -m src.ranking \
          --model "$model" --dataset "$dataset" \
          --n_samples "$n" --fraction_of_data_to_use "$frac" \
          --self_certainty --seed "$seed"
      done
    done
  done
done

echo "=== Scaling sweep done — see results/ranking_logs.tsv ==="
