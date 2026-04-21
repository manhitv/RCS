
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import evaluate
import os
import pickle
from tqdm import tqdm
import torch
import config
import json
import argparse
import numpy as np
from collections import Counter
import pandas as pd

import logging
logging.basicConfig(level=logging.ERROR)

from sentence_transformers import SentenceTransformer
from utils import (
    MODEL_PATH_DICT,
    set_seed,
    compute_self_certainty_scores, 
    get_self_certainty_sample, 
    compute_weighted_mean, 
    clean_generation, 
    clean_answer, 
    is_correct,
    compute_label,
    extract_math_response,
    code_eval
    )

from ranking_modex import modex_select


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


# def compute_medoid(points):
#     """
#     points: (N, D)
#     returns: (D,)
#     """
#     # pairwise distance matrix (N x N)
#     dists = torch.cdist(points, points, p=2)
    
#     # sum distance per point
#     total_dist = dists.sum(dim=1)
    
#     # index of medoid
#     medoid_idx = torch.argmin(total_dist)
    
#     return points[medoid_idx]

def compute_medoid(points, weights=None):
    """
    points: (N, D) hoặc (D,)
    weights: (N,) or None
    returns: (D,)
    """

    # 🔴 Case 1: chỉ 1 vector (D,)
    if points.dim() == 1:
        return points

    # 🔴 Case 2: chỉ 1 sample (1, D)
    if points.size(0) == 1:
        return points[0]

    dists = torch.cdist(points, points, p=2)  # (N, N)

    if weights is not None:
        # đảm bảo shape đúng
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
        # 1. Base (Uniform / Mean)
        # =========================
        rds_base_center = torch.mean(embeddings, dim=0)
        rds_base = compute_rds(embeddings, rds_base_center)

        # =========================
        # 2. Frequency-weighted
        # =========================
        freq_counts = Counter(texts_for_embedding)
        probs_freq = np.array(
            [freq_counts[t] for t in texts_for_embedding],
            dtype=np.float32
        )
        probs_freq /= probs_freq.sum()

        rds_freq_center = compute_weighted_mean(
            embeddings,
            torch.tensor(probs_freq, dtype=torch.float32, device=device)
        )

        rds_freq = compute_rds(embeddings, rds_freq_center)


        # =========================
        # 3. Probability-weighted
        # =========================
        probs_prob = np.exp(-np.array(samples_avg_nll))
        probs_prob /= probs_prob.sum()

        rds_prob_center = compute_weighted_mean(
            embeddings,
            torch.tensor(probs_prob, dtype=torch.float32, device=device)
        )

        rds_prob = compute_rds(embeddings, rds_prob_center)


        # =========================
        # 4. Medoid (L2)
        # =========================
        rds_medoid_center_base = compute_medoid(embeddings)
        rds_medoid = compute_rds(embeddings, rds_medoid_center_base)
        
        rds_medoid_center_freq = compute_medoid(embeddings, weights=torch.tensor(probs_freq, dtype=torch.float32, device=device))
        rds_medoid_freq = compute_rds(embeddings, rds_medoid_center_freq)
        
        rds_medoid_center_prob = compute_medoid(embeddings, weights=torch.tensor(probs_prob, dtype=torch.float32, device=device))
        rds_medoid_prob = compute_rds(embeddings, rds_medoid_center_prob)
        
        if args.dataset in ['gsm8k', 'arith_long'] and args.raw_answers:
            rds_raw_base = compute_rds_raw(extracted_answers)
            rds_raw_freq = compute_rds_raw(extracted_answers, weights=probs_freq)
            rds_raw_prob = compute_rds_raw(extracted_answers, weights=probs_prob)
            rds_raw_medoid = compute_rds_raw_medoid_weighted(extracted_answers)
            rds_raw_medoid_freq = compute_rds_raw_medoid_weighted(extracted_answers, weights=probs_freq)
            rds_raw_medoid_prob = compute_rds_raw_medoid_weighted(extracted_answers, weights=probs_prob)
        
        # --- Ranking and find samples ---
        if len(samples_nll) == 0:
            nll_sample = None
            avg_nll_sample = None
            rds_base_sample = None
            rds_freq_sample = None
            rds_prob_sample = None
            rds_medoid_sample = None
            rds_medoid_freq_sample = None
            rds_medoid_prob_sample = None
            majority_sample = None
        else:
            if args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med', 'mmlu_pro', 'crux_eval']:

                nll_sample = extracted_answers[np.argmin(samples_nll)]
                avg_nll_sample = extracted_answers[np.argmin(samples_avg_nll)]
                rds_base_sample = extracted_answers[torch.argmin(rds_base).item()]
                rds_freq_sample = extracted_answers[torch.argmin(rds_freq).item()]
                rds_prob_sample = extracted_answers[torch.argmin(rds_prob).item()]
                rds_medoid_sample = extracted_answers[torch.argmin(rds_medoid).item()]
                rds_medoid_freq_sample = extracted_answers[torch.argmin(rds_medoid_freq).item()]
                rds_medoid_prob_sample = extracted_answers[torch.argmin(rds_medoid_prob).item()]
                
                if args.include_oracle:
                    oracle_sample = "None"
                    for idx, ans in enumerate(extracted_answers):
                        acc = evaluation_sample(
                            dataset=args.dataset,
                            text=ans,
                            answer=gen["answer"],
                            question=gen["question"] if "question" in gen else None,
                            rouge=rouge,
                            api_type=args.api_type,
                            eval_method=args.eval_method,
                            threshold=args.threshold
                        )
                        
                        if acc == 1:
                            oracle_sample = ans
                            break
                
                if args.dataset in ['gsm8k', 'arith_long'] and args.raw_answers:
                    rds_raw_base_sample = extracted_answers[np.argmin(rds_raw_base)]
                    rds_raw_freq_sample = extracted_answers[np.argmin(rds_raw_freq)]
                    rds_raw_prob_sample = extracted_answers[np.argmin(rds_raw_prob)]
                    rds_raw_medoid_sample = extracted_answers[np.argmin(rds_raw_medoid)]
                    rds_raw_medoid_freq_sample = extracted_answers[np.argmin(rds_raw_medoid_freq)]
                    rds_raw_medoid_prob_sample = extracted_answers[np.argmin(rds_raw_medoid_prob)]

                freq = Counter(extracted_answers)
                majority_sample = freq.most_common(1)[0][0]
            
            else:
            
                nll_sample = cleaned_texts[np.argmin(samples_nll)]
                avg_nll_sample = cleaned_texts[np.argmin(samples_avg_nll)]
                rds_base_sample = cleaned_texts[torch.argmin(rds_base).item()]
                rds_freq_sample = cleaned_texts[torch.argmin(rds_freq).item()]
                rds_prob_sample = cleaned_texts[torch.argmin(rds_prob).item()]
                rds_medoid_sample = cleaned_texts[torch.argmin(rds_medoid).item()]
                rds_medoid_freq_sample = cleaned_texts[torch.argmin(rds_medoid_freq).item()]
                rds_medoid_prob_sample = cleaned_texts[torch.argmin(rds_medoid_prob).item()]
                
                if args.dataset in ['gsm8k', 'arith_long'] and args.raw_answers:
                    rds_raw_base_sample = cleaned_texts[np.argmin(rds_raw_base)]
                    rds_raw_freq_sample = cleaned_texts[np.argmin(rds_raw_freq)]
                    rds_raw_prob_sample = cleaned_texts[np.argmin(rds_raw_prob)]
                    rds_raw_medoid_sample = cleaned_texts[np.argmin(rds_raw_medoid)]
                    rds_raw_medoid_freq_sample = cleaned_texts[np.argmin(rds_raw_medoid_freq)]
                    rds_raw_medoid_prob_sample = cleaned_texts[np.argmin(rds_raw_medoid_prob)]

                if args.include_oracle:
                    oracle_sample = "None"
                    for idx, ans in enumerate(cleaned_texts):
                        acc = evaluation_sample(
                            dataset=args.dataset,
                            text=ans,
                            answer=gen["answer"],
                            question=gen["question"] if "question" in gen else None,
                            rouge=rouge,
                            api_type=args.api_type,
                            eval_method=args.eval_method,
                            threshold=args.threshold
                        )
                        
                        if acc == 1:
                            oracle_sample = ans
                            break

                freq = Counter(cleaned_texts)
                majority_sample = freq.most_common(1)[0][0]
        
        if args.self_certainty:
            sc_cache_path = f"{config.output_dir}/{args.dataset}_{args.model}_N={args.n_samples}_F={args.fraction_of_data_to_use}_A={args.api_type}_S={args.seed}__self_certainty.pkl"
            os.makedirs(os.path.dirname(sc_cache_path), exist_ok=True)

            if os.path.exists(sc_cache_path):
                with open(sc_cache_path, "rb") as f:
                    all_self_certainty = pickle.load(f)
            else:
                prompts = [gen["prompt"] for gen in generations]
                generated_texts_list = [gen["cleaned_generated_texts"] for gen in generations]
                
                all_self_certainty = compute_self_certainty_scores(
                    model_dir=MODEL_PATH_DICT[args.model],
                    prompts=prompts,
                    generated_texts_list=generated_texts_list,
                    batch_size=4,
                    device=device
                )
                
                with open(sc_cache_path, "wb") as f:
                    pickle.dump(all_self_certainty, f)
                print(f"Saved self-certainty scores to {sc_cache_path}")
                
            gen["samples_ce"] = all_self_certainty[i]
        
        if args.modex:
            modex_idx = modex_select(cleaned_texts, adjacency='text', tau=0.8, goodness_of_cut='conductance', emb_encoder=embed_model)
            if args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med', 'mmlu_pro', 'crux_eval']:
                modex_sample = extracted_answers[modex_idx]
            else:
                modex_sample = cleaned_texts[modex_idx]
        
        # --- Self-certainty sample ---
        if "samples_ce" in gen:
            sc_scores = np.array(gen["samples_ce"])
            sc_scores = [score for idx, score in enumerate(sc_scores) if idx not in blank_indices] if args.ignore_null else sc_scores
            
            if len(sc_scores) == 0:
                self_certainty_sample = None
            else:
                if args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med', 'mmlu_pro', 'crux_eval']:
                    self_certainty_sample = get_self_certainty_sample(sc_scores, extracted_answers)
                else:
                    self_certainty_sample = get_self_certainty_sample(sc_scores, cleaned_texts)
        else:
            self_certainty_sample = None

        # --- Evaluation ---
        method_base = ['nll', 'avg_nll', 'rds_base', 'rds_freq', 'rds_prob', 'rds_medoid', 'rds_medoid_freq', 'rds_medoid_prob', 'majority']
        sample_base = [nll_sample, avg_nll_sample, rds_base_sample, rds_freq_sample, rds_prob_sample, rds_medoid_sample, rds_medoid_freq_sample, rds_medoid_prob_sample, majority_sample]
        
        if args.dataset in ['gsm8k', 'arith_long'] and args.raw_answers:
            method_base += ['rds_raw_base', 'rds_raw_freq', 'rds_raw_prob', 'rds_raw_medoid', 'rds_raw_medoid_freq', 'rds_raw_medoid_prob']
            sample_base += [rds_raw_base_sample, rds_raw_freq_sample, rds_raw_prob_sample, rds_raw_medoid_sample, rds_raw_medoid_freq_sample, rds_raw_medoid_prob_sample]
        
        if args.include_oracle:
            method_base += ['oracle']
            sample_base += [oracle_sample]
        
        # =========================
        # NEW BASELINES
        # =========================
        if args.self_certainty:
            method_base += ['self_certainty']
            sample_base += [self_certainty_sample]
        
        if args.modex:
            method_base += ['modex']
            sample_base += [modex_sample]
        
        methods = method_base
        samples = sample_base    
        # methods = method_base if self_certainty_sample is None else method_base + ['self_certainty']
        # samples = sample_base if self_certainty_sample is None else sample_base + [self_certainty_sample]

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
    tsv_file = f'results/ranking_logs.tsv'
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
    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
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