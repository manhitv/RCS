# RCS

Source code for the paper: **"Beyond Majority Voting: Efficient Best-Of-N with Radial Consensus Score"** 

- Preprint: Comming Soon

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
python generation.py \
    --model $model \
    --dataset $dataset \
    --n_samples $n \
    --fraction_of_data_to_use $fraction \
    --max_new_tokens $max_new_tokens \
    --seed $seed
```


## 3. Best-Of-N

```bash
python ranking.py \
    --model $model \
    --dataset $dataset \
    --n_samples $n \
    --self_certainty \
    --fraction_of_data_to_use $fraction \
    --threshold $threshold \
    --include_oracle \
    --seed $seed 
```

## 4. Analysis

```bash
# Clean-answer setting
python ranking.py --ignore_null

# Full reasoning path
python ranking.py --full_answers
```

## Parameters: 
* `--model`: Model identifier (e.g., `qwen2.5-7b`, `llama3.1-8b`)
* `--dataset`: Dataset name
* `--n_samples`: Number of generations per input
* `--self_certainty`: Optionally include Self-certainty baseline
* `--fraction_of_data_to_use`: Fraction of the dataset to use (1 = full dataset)
* `--threshold`: correctness threshold for short-form QA (SciQ, NQ). Default=0.3
* `--include_oracle`: Optionally include Oracle baseline