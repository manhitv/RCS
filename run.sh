python -m src.generation --model qwen2.5-3b --dataset gpqa --n_samples 10 --fraction_of_data_to_use 1 --seed 42

python -m src.ranking --model qwen2.5-3b --dataset gpqa --n_samples 10 --fraction_of_data_to_use 1 --self_certainty --modex  --include_oracle --seed 42