import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import evaluate
import logging
logging.basicConfig(level=logging.ERROR)

from vllm import LLM

from . import config
from .utils import (
    MODEL_PATH_DICT,
    set_seed, 
    parse_dataset,
    generate_sequences
    )

# --------------------
# MAIN PIPELINE
# --------------------
def main(args):
    experiment_id = os.getpid()
    cache_dir = f"/tmp/rouge_cache_{experiment_id}"
    os.environ['HF_EVALUATE_CACHE'] = cache_dir
    rouge = evaluate.load('rouge', experiment_id=experiment_id, cache_dir=cache_dir)
    
    # Load dataset
    dataset = parse_dataset(args=args)

    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset[: int(len(dataset) * args.fraction_of_data_to_use)]

    # Init model
    hf_model_dir = MODEL_PATH_DICT[args.model]
    llm = LLM(model=hf_model_dir, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=2048) # 0.9 by default, 0.5 for GPT-OSS-20B
    print(f"Loaded model {args.model} for generation.")
    
    # Check if output already exists
    output_path = f"{config.output_dir}/{args.dataset}_{args.model}_N={args.n_samples}_F={args.fraction_of_data_to_use}_A={args.api_type}_S={args.seed}__generation.pkl"
    if os.path.exists(output_path):
        print(f"Output already exists at {output_path}. Ignore generation results...")
        with open(output_path, "rb") as f:
            sequences = pickle.load(f)
        return sequences
    
    # Run generation
    sequences = generate_sequences(llm=llm, dataset=dataset, rouge=rouge, args=args)
    
    # Save
    # output_path = f"{config.output_dir}/{args.dataset}_{args.model}_N={args.n_samples}_F={args.fraction_of_data_to_use}_A={args.api_type}_S={args.seed}__generation.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(sequences, f)

    print(f"Saved results to {output_path}")
    return sequences

# --------------------
# CLI ENTRY
# --------------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
    parser.add_argument('--few_shot_num', type=int, default=5)
    parser.add_argument('--model', type=str, default='gemma-7b', required=True)
    parser.add_argument('--dataset', type=str, default='coqa', required=True)
    parser.add_argument('--max_new_tokens', type=int, default=32) # 512 for gsm8k
    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
    parser.add_argument('--api_type', type=str, default='cohere', choices=['gemini', 'cohere'], help='API type for LLM evaluation')
    
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)