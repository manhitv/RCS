import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import evaluate
import pickle
from tqdm import tqdm
import torch
import argparse
import numpy as np
from collections import Counter
import pandas as pd

import logging
logging.basicConfig(level=logging.ERROR)

from sentence_transformers import SentenceTransformer

from . import config
from .utils import (
    MODEL_PATH_DICT,
    set_seed,
    compute_self_certainty_scores, 
    get_self_certainty_sample, 
    compute_weighted_mean, 
    compute_label,
    code_eval,
    modex_select
    )

def evaluation_sample(dataset, text, answer, rouge, question=None, eval_method="rougeL", api_type="cohere", threshold=0.3):
    
    # if dataset in ['gsm8k']:
    #     text = extract_math_response(text=text, args=args)
    # else:
    #     text = clean_generation(text)
        
    if dataset in ['svamp', 'arith']: # exact match for math datasets
        eval_score = compute_label(generation=text, ground_truth=answer, eval_method="exact_match")
    elif dataset in ['gsm8k', 'arith_long']: # exact match after rounding to 1 decimal place for math datasets
        eval_score = int(text == np.round(answer, 1))
        # model_answer = clean_answer(text)
        # eval_score = is_correct(model_answer=model_answer, answer=answer)
    elif dataset in ['formal_logic', 'pro_med', 'mmlu_pro']:
        eval_score = int(text == answer)
        
    elif dataset in ['crux_eval']:
        eval_score = code_eval(text, answer)
    
    else:
        eval_score = compute_label(generation=text, ground_truth=answer, question=question, eval_method=eval_method, rouge=rouge, api_type=api_type)
    
    if dataset in ['gsm8k', 'svamp', 'arith', 'arith_long', 'formal_logic', 'pro_med', 'mmlu_pro', 'crux_eval']:
        acc = int(eval_score == 1.0)
    else:
        if eval_method == "rougeL":
            acc = int(eval_score > threshold)
        else:
            acc = int(eval_score)
        
    return acc


def geometric_median(points, eps=1e-5, max_iter=100):
    """
    points: (N, D) tensor
    returns: (D,) tensor
    """
    median = points.mean(dim=0)  # init
    
    for _ in range(max_iter):
        diffs = points - median
        distances = torch.norm(diffs, p=2, dim=1).clamp(min=eps) 
        
        weights = 1.0 / distances
        new_median = (points * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
        
        if torch.norm(new_median - median) < eps:
            break
        median = new_median
    
    return median


def compute_medoid(points, weights=None):
    """
    points: (N, D) hoặc (D,)
    weights: (N,) or None
    returns: (D,)
    """

    if points.dim() == 1:
        return points

    if points.size(0) == 1:
        return points[0]

    dists = torch.cdist(points, points, p=2)  # (N, N)

    if weights is not None:
        weights = weights.view(1, -1)  # (1, N)
        total_dist = (dists * weights).sum(dim=1)
    else:
        total_dist = dists.sum(dim=1)

    medoid_idx = torch.argmin(total_dist)
    return points[medoid_idx]


# =========================
# Helpers
# =========================
def compute_rds(embeddings, center):
    diffs = embeddings - center.unsqueeze(0)
    return torch.norm(diffs, p=2, dim=-1)

def compute_rds_raw(answers, weights=None):
    answers = np.array(answers)
    mean_answer = np.average(answers, weights=weights)
    return np.abs(answers - mean_answer)

def compute_rds_raw_medoid_weighted(answers, weights=None):
    answers = np.array(answers)
    N = len(answers)
    
    if weights is None:
        weights = np.ones(N)
    else:
        weights = np.array(weights)
    
    # pairwise absolute distances (N x N)
    dists = np.abs(answers[:, None] - answers[None, :])
    
    # weighted sum of distances
    total_dist = (dists * weights[None, :]).sum(axis=1)
    
    # medoid index
    medoid_idx = np.argmin(total_dist)
    medoid = answers[medoid_idx]
    
    # return distance to medoid
    return np.abs(answers - medoid)


def main(args):
    experiment_id = os.getpid()
    cache_dir = f"/tmp/rouge_cache_{experiment_id}"
    os.environ['HF_EVALUATE_CACHE'] = cache_dir
    rouge = evaluate.load('rouge', experiment_id=experiment_id, cache_dir=cache_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer(args.embed_model).to(device)

    # --- Load generations ---
    gen_path = f"{config.output_dir}/{args.dataset}_{args.model}_N={args.n_samples}_F={args.fraction_of_data_to_use}_A={args.api_type}_S={args.seed}__generation.pkl"
    with open(gen_path, "rb") as infile:
        generations = pickle.load(infile)

    accuracy = {"greedy": []}
    
    for i, gen in enumerate(tqdm(generations, desc="Processing generations")):
        
        # --- Find the least uncertain samples ---
        cleaned_texts = gen["cleaned_generated_texts"]
        extracted_answers = gen["extracted_answers"] if "extracted_answers" in gen else [None] * len(cleaned_texts)
        samples_avg_nll = gen["samples_avg_nll"]
        samples_nll = gen["samples_nll"]
        
        if args.ignore_null:
            blank_indices = [idx for idx, ans in enumerate(extracted_answers) if ans in [None, ""]]
            cleaned_texts = [text for idx, text in enumerate(cleaned_texts) if idx not in blank_indices]
            extracted_answers = [ans for idx, ans in enumerate(extracted_answers) if idx not in blank_indices]
            samples_avg_nll = [nll for idx, nll in enumerate(samples_avg_nll) if idx not in blank_indices]
            samples_nll = [nll for idx, nll in enumerate(samples_nll) if idx not in blank_indices]
        
        # --- RDS score ---
        if args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med', 'mmlu_pro', 'crux_eval']:
            if args.full_answers:
                texts_for_embedding = cleaned_texts
            else:
                texts_for_embedding = [str(j) for j in extracted_answers]
        else:
            texts_for_embedding = cleaned_texts
            
        embeddings = embed_model.encode(texts_for_embedding, convert_to_tensor=True, device=device)


        # =========================
        # Cosine-similarity-weighted (SCW baseline + RCS_cosine)
        # =========================
        full_text_embeddings = embed_model.encode(cleaned_texts, convert_to_tensor=True, device=device)
        norms = torch.norm(full_text_embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        normed_embeddings = full_text_embeddings / norms
        cosine_sim_matrix = torch.mm(normed_embeddings, normed_embeddings.t())  # (N, N)
        sim_sum = cosine_sim_matrix.sum(dim=1)  # (N,): total cosine similarity of each sample to all others

        # SSC: group cosine-similarity sums by unique answer text, pick the highest-scoring group.
        # This is a discrete voting step (like majority voting but cosine-weighted), so it can still
        # miss minority correct answers. Representative = member of the winning group with highest sim_sum.
        answer_to_indices = {}
        for idx, text in enumerate(cleaned_texts):
            answer_to_indices.setdefault(text, []).append(idx)
        answer_group_scores = {
            text: sum(sim_sum[idx].item() for idx in indices)
            for text, indices in answer_to_indices.items()
        }
        scw_best_text = max(answer_group_scores, key=answer_group_scores.get)
        scw_idx = max(answer_to_indices[scw_best_text], key=lambda idx: sim_sum[idx].item())

        # RCS_cosine: use per-sample sim_sum as continuous Fréchet-mean weights.
        # No grouping/voting step — all candidates ranked by distance to the weighted center.
        probs_cosine = sim_sum / sim_sum.sum().clamp(min=1e-8)
        rds_cosine_center = compute_weighted_mean(embeddings, probs_cosine)
        rds_cosine = compute_rds(embeddings, rds_cosine_center)

        # --- Sample selection ---
        if len(samples_nll) == 0:
            scw_sample = None
            rds_cosine_sample = None
        else:
            if args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med', 'mmlu_pro', 'crux_eval']:
                scw_sample = extracted_answers[scw_idx]
                rds_cosine_sample = extracted_answers[torch.argmin(rds_cosine).item()]
            else:
                scw_sample = cleaned_texts[scw_idx]
                rds_cosine_sample = cleaned_texts[torch.argmin(rds_cosine).item()]

        methods = ['scw', 'rds_cosine']
        samples = [scw_sample, rds_cosine_sample]

        for method, sample in zip(methods, samples):
            acc = evaluation_sample(
                dataset=args.dataset,
                text=sample,
                answer=gen["answer"],
                question=gen["question"] if "question" in gen else None,
                rouge=rouge,
                api_type=args.api_type,
                eval_method=args.eval_method,
                threshold=args.threshold
            )
            
            
            if method not in accuracy:
                accuracy[method] = []
                
            accuracy[method].append(acc)
            
            # For debug
            if i < 3:
                print(f"Sample {i} | Method: {method} | Acc: {acc} | Sample: {sample}...")

        # Greedy acc
        greedy_acc = evaluation_sample(
            dataset=args.dataset,
            text=gen['greedy_text'],
            answer=gen["answer"],
            question=gen["question"] if "question" in gen else None,
            rouge=rouge,
            api_type=args.api_type,
            eval_method=args.eval_method,
            threshold=args.threshold
        )
        accuracy["greedy"].append(greedy_acc)
        
    # --- Reporting ---
    results = {}
    print("\n=== Metric Performance ===")
    for method, values in accuracy.items():
        final_acc = np.mean(values)
        results[method] = round(final_acc, 4)
        print(f"{method:55s} → ACC: {final_acc:.4f}")

    # --- Prepare row to append ---
    row = {
        "timestamp": args.timestamp,
        "dataset": args.dataset,
        "model": args.model,
        "embed_model": args.embed_model,
        "n_samples": args.n_samples,
        "threshold": args.threshold,
        "eval_method": args.eval_method,
        "api_type": args.api_type,
        "fraction_of_data_to_use": args.fraction_of_data_to_use,
        "ignore_null": args.ignore_null,
        "raw_answers": args.raw_answers,
        "full_answers": args.full_answers,
        "include_oracle": args.include_oracle,
        "seed": args.seed,
    }
    row.update(results)
    new_row_df = pd.DataFrame([row])

    # --- Check if file exists ---
    result_dir = 'results'
    # os.mkdir(result_dir, exist_ok=True)
    
    tsv_file = f'{result_dir}/ranking_logs__ssc.tsv'
    
    if os.path.exists(tsv_file):
        df = pd.read_csv(tsv_file, sep='\t')
        
        # Union all columns
        all_cols = sorted(set(df.columns).union(new_row_df.columns))

        df = df.reindex(columns=all_cols)
        new_row_df = new_row_df.reindex(columns=all_cols)

        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        df = new_row_df

    # --- Save back ---
    df.to_csv(tsv_file, sep='\t', index=False)

    return results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Best-of-N accuracy for different ranking methods')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--embed_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--self_certainty', action='store_true', help='Whether to compute self-certainty scores')
    parser.add_argument('--modex', action='store_true', help='Whether to compute DeepConf scores')
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0, help='Fraction of data to use for evaluation (for quick testing)')
    parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for binary classification of correctness (used for non-math datasets)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--eval_method', type=str, default='rougeL', help='Evaluation method for non-math datasets (e.g., rougeL or api)')
    parser.add_argument('--api_type', type=str, default='cohere', choices=['gemini', 'cohere'], help='API type for LLM evaluation')
    parser.add_argument('--ignore_null', action='store_true', help='Whether to ignore samples with null generations')
    parser.add_argument('--raw_answers', action='store_true', help='Whether to compute RDS variants on raw answers instead of embeddings (only for math datasets)')
    parser.add_argument('--full_answers', action='store_true', help='Whether to compute RDS variants on full generated texts instead of extracted answers (only for math datasets)')
    parser.add_argument('--include_oracle', action='store_true', help='Whether to include oracle method (only for math datasets)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    args.timestamp = timestamp

    print(f"RANKING: Dataset={args.dataset}, Model={args.model}, N={args.n_samples}, F={args.fraction_of_data_to_use}, T={args.threshold}, S={args.seed}, E={args.eval_method}, A={args.api_type}, I={args.ignore_null}.")
    set_seed(args.seed)
    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    print(f"Total evaluation time: {end_time - start_time}")