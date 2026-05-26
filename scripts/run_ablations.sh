#!/usr/bin/env bash
# Ablation studies (Section 4.2 / Appendix).
#   1. Embedding-model sensitivity (MiniLM | mpnet | roberta-large)
#   2. Correctness-threshold sweep on SciQ / GPQA (tau = 0.1..0.5)
#   3. Full-trajectory embeddings (--full_answers) — embedding-collapse check
#   4. Clean-answer setting (--ignore_null)
#
# Generation must already be cached (run scripts/run_main_table.sh first).

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=("qwen2.5-3b" "llama3.2-3b")
SEEDS=(42 44 46)
N=10

# ------------------------------------------------------------------
# 1. Embedding-model sensitivity (Figure embed_*)
# ------------------------------------------------------------------
echo "### Ablation 1/4 — embedding-model sensitivity"
EMBED_MODELS=("all-MiniLM-L6-v2" "all-mpnet-base-v2" "all-roberta-large-v1")
EMBED_DATASETS=("arith_long" "formal_logic")

declare -A FRACTION_EMB=(
    [arith_long]=0.2
    [formal_logic]=1
)

for seed in "${SEEDS[@]}"; do
  for dataset in "${EMBED_DATASETS[@]}"; do
    frac=${FRACTION_EMB[$dataset]}
    for model in "${MODELS[@]}"; do
      for embed in "${EMBED_MODELS[@]}"; do
        python -m src.ranking \
          --model "$model" --dataset "$dataset" \
          --n_samples "$N" --fraction_of_data_to_use "$frac" \
          --embed_model "$embed" --seed "$seed"
      done
    done
  done
done

# ------------------------------------------------------------------
# 2. Correctness-threshold sweep (Figure threshold_*)
# ------------------------------------------------------------------
echo "### Ablation 2/4 — correctness-threshold sweep"
THRESHOLDS=(0.1 0.2 0.3 0.4 0.5)
THRESHOLD_DATASETS=("sciq" "gpqa")

for seed in "${SEEDS[@]}"; do
  for dataset in "${THRESHOLD_DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for th in "${THRESHOLDS[@]}"; do
        python -m src.ranking \
          --model "$model" --dataset "$dataset" \
          --n_samples "$N" --fraction_of_data_to_use 1 \
          --threshold "$th" --eval_method rougeL --seed "$seed"
      done
    done
  done
done

# ------------------------------------------------------------------
# 3. Full-trajectory embeddings (embedding-collapse check)
# ------------------------------------------------------------------
echo "### Ablation 3/4 — full-trajectory embeddings (--full_answers)"
FULL_DATASETS=("arith_long" "formal_logic" "gsm8k")
declare -A FRACTION_FULL=(
    [arith_long]=0.2
    [formal_logic]=1
    [gsm8k]=0.15
)
for seed in "${SEEDS[@]}"; do
  for dataset in "${FULL_DATASETS[@]}"; do
    frac=${FRACTION_FULL[$dataset]}
    for model in "${MODELS[@]}"; do
      python -m src.ranking \
        --model "$model" --dataset "$dataset" \
        --n_samples "$N" --fraction_of_data_to_use "$frac" \
        --full_answers --seed "$seed"
    done
  done
done

# ------------------------------------------------------------------
# 4. Clean-answer setting (drop blank/null extracted answers)
# ------------------------------------------------------------------
echo "### Ablation 4/4 — clean-answer setting (--ignore_null)"
CLEAN_DATASETS=("arith_long" "formal_logic")
declare -A FRACTION_CLEAN=(
    [arith_long]=0.2
    [formal_logic]=1
)
for seed in "${SEEDS[@]}"; do
  for dataset in "${CLEAN_DATASETS[@]}"; do
    frac=${FRACTION_CLEAN[$dataset]}
    for model in "${MODELS[@]}"; do
      python -m src.ranking \
        --model "$model" --dataset "$dataset" \
        --n_samples "$N" --fraction_of_data_to_use "$frac" \
        --ignore_null --seed "$seed"
    done
  done
done

echo "=== Ablations done — see results/ranking_logs.tsv ==="
