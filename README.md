<div align="center">
<img src="assets/rcs_logo.svg" height=140 alt="RCS">
  <h1><b> RCS: Radial Consensus Score </b></h1>
  <p><i>Efficient Best-of-N — pick the most consistent answer, not the luckiest one.</i></p>
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2604.12196-b31b1b)](https://arxiv.org/pdf/2604.12196)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/Powered_by-Transformers-yellow)](https://github.com/huggingface/transformers)
[![vLLM](https://img.shields.io/badge/Powered_by-vLLM-orange)](https://github.com/vllm-project/vllm)

</div>

Source code for the paper **"Efficient Best-Of-N with Radial Consensus Score"**.

> 📜 **Preprint:** Coming soon.

---

## 1. Environment Setup

Create a conda environment from the provided YAML file:

```bash
conda env create -f environment.yaml
conda activate <env_name>
```
* Before running any scripts, make sure to update file paths in `config.py` according to your local directory structure.

## 2. Generation

```bash
python -m src.generation \
    --model $model \
    --dataset $dataset \
    --n_samples $n \
    --fraction_of_data_to_use $fraction \
    --max_new_tokens $max_new_tokens \
    --seed $seed
```


## 3. Best-Of-N

```bash
python -m src.ranking \
    --model $model \
    --dataset $dataset \
    --n_samples $n \
    --self_certainty \
    --modex \
    --fraction_of_data_to_use $fraction \
    --threshold $threshold \
    --include_oracle \
    --seed $seed 
```

## 4. Analysis

```bash
# Blackbox setting
python -m src.blackbox --client cohere --dataset bbh_date

# Embed models: all-roberta-large-v1 | all-mpnet-base-v2
python -m src.ranking --embed_model all-roberta-large-v1

# Clean-answer setting
python -m src.ranking --ignore_null

# Full reasoning path
python -m src.ranking --full_answers
```

## Parameters: 
* `--model`: Model identifier (e.g., `qwen2.5-7b`, `llama3.1-8b`)
* `--dataset`: Dataset name (e.g., `gpqa`, `formal_logic`)
* `--n_samples`: Number of generations per input
* `--self_certainty`, `--modex`: Optionally include Self-certainty/ModeX baselines
* `--fraction_of_data_to_use`: Fraction of the dataset to use (1 = full dataset)
* `--threshold`: correctness threshold for short-form QA (SciQ, NQ). Default=0.3.
* `--include_oracle`: Optionally include Oracle baseline

## Example
```bash
bash run.sh
```