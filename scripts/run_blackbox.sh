#!/usr/bin/env bash
# Black-box evaluation (Table 4a): Cohere `command-a-03-2025` on
# GPQA, MMLU-Pro, BBH-Date, BBH-Navigate.
#
# Requires:
#   export COHERE_API_KEY="..."

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${COHERE_API_KEY:-}" ]]; then
  echo "ERROR: COHERE_API_KEY is not set." >&2
  exit 1
fi

N="${N:-10}"
TEMPERATURE="${TEMPERATURE:-1}"
TOP_P="${TOP_P:-0.99}"
MAX_TOKENS="${MAX_TOKENS:-512}"

DATASETS=("gpqa" "mmlu_pro" "bbh_date" "bbh_nav")

for dataset in "${DATASETS[@]}"; do

  echo "=== black-box | dataset=$dataset N=$N T=$TEMPERATURE p=$TOP_P max_tokens=$max_tokens ==="
  python -m src.blackbox \
    --client cohere --dataset "$dataset" \
    --n_samples "$N" --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" --max_tokens "$max_tokens"
done

echo "=== Black-box sampling done — see results/cohere_*.csv ==="
