
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
    compute_weighted_mean, 
    clean_generation, 
    clean_answer, 
    is_correct,
    compute_label,
    extract_math_response
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
    elif dataset in ['formal_logic', 'pro_med']:
        eval_score = int(text == answer)
    else:
        eval_score = compute_label(generation=text, ground_truth=answer, question=question, eval_method=eval_method, rouge=rouge, api_type=api_type)
    
    if dataset in ['gsm8k', 'svamp', 'arith', 'arith_long', 'formal_logic', 'pro_med']:
        acc = int(eval_score == 1.0)
    else:
        if eval_method == "rougeL":
            acc = int(eval_score > threshold)
        else:
            acc = int(eval_score)
        
    return acc


import numpy as np
from sentence_transformers import SentenceTransformer

def modex_select(texts, adjacency='text', tau=0.8, goodness_of_cut='conductance', emb_encoder=None):
    """
    ModeX selection - Chọn response tốt nhất mà không cần evaluator.
    
    Args:
        texts (list[str]): Danh sách các câu trả lời (các candidate responses)
        adjacency (str): 'semantics' (mặc định, dùng embedding), 'text', hoặc 'both'
        tau (float): Ngưỡng dừng spectral cut (thường 0.3 ~ 0.6, càng cao càng ít cắt)
        goodness_of_cut (str): 'conductance', 'cutratio', hoặc 'ngc'
    
    Returns:
        int: Index của response được chọn (modal / best-of-N)
    """
    if len(texts) <= 1:
        return 0

    n = len(texts)
    agent_names = [f"agent_{i}" for i in range(n)]
    agent_responses = dict(zip(agent_names, texts))

    # === 1. Tính adjacency matrix ===
    if adjacency == 'semantics':
        A = compute_semantics_adjacency_matrix(agent_names, agent_responses, emb_encoder)
    elif adjacency == 'text':
        A = compute_text_adjacency_matrix(agent_names, agent_responses)
    elif adjacency == 'both':
        A_sem = compute_semantics_adjacency_matrix(agent_names, agent_responses, emb_encoder)
        A_text = compute_text_adjacency_matrix(agent_names, agent_responses)
        A = 0.5 * (A_sem + A_text)
    else:
        raise ValueError("adjacency phải là 'semantics', 'text' hoặc 'both'")

    # === 2. Thực hiện recursive spectral graph cut ===
    _A = A.copy()
    current_names = agent_names.copy()

    # max_iter = 10
    # iter_count = 0
    while True:
        # Spectral clustering (2-way cut)
        info = graph_cut(_A, current_names)   # hàm này mình sẽ định nghĩa bên dưới
        # iter_count += 1
        # if iter_count > max_iter:
        #     break
        
        # Chọn nhóm lớn hơn
        g1 = info['groups']['group_1']
        g2 = info['groups']['group_2']
        group = g1 if len(g1) >= len(g2) else g2
        n_group = g2 if len(g1) >= len(g2) else g1

        prev_n = len(current_names)
        group_indices = [current_names.index(name) for name in group]
        n_group_indices = [current_names.index(name) for name in n_group]

        # Tính goodness of cut
        phi = goodness_of_cut_func(_A, group_indices, n_group_indices, goodness_of_cut)

        # Nếu cắt không còn tốt nữa (phi >= tau) → dừng và chọn node có degree cao nhất
        if phi >= tau:
            degrees = np.sum(_A, axis=1)
            best_idx = int(np.argmax(degrees))
            best_name = current_names[best_idx]
            return agent_names.index(best_name)   # trả về index gốc

        # Cắt tiếp nhóm lớn hơn
        _A = _A[np.ix_(group_indices, group_indices)]
        current_names = [current_names[i] for i in group_indices]

        if len(current_names) == prev_n or len(current_names) <= 1:
            break

    # Fallback: chọn ngẫu nhiên 1 cái trong cluster cuối
    selected_name = np.random.choice(current_names)
    return agent_names.index(selected_name)


# ====================== Các hàm hỗ trợ ======================

def compute_semantics_adjacency_matrix(agent_names, agent_responses, emb_encoder):
    texts = [str(agent_responses[name]) for name in agent_names]
    embeddings = emb_encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return np.dot(embeddings, embeddings.T)


# def compute_text_adjacency_matrix(agent_names, agent_responses):
#     """Jaccard similarity trên unigram + bigram + trigram (gộp lại)"""
#     n = len(agent_names)
#     A = np.zeros((n, n))

#     def get_ngrams(text, ngram=3):
#         tokens = str(text).lower().split()
#         ngrams = set()
#         for k in range(1, ngram + 1):
#             for i in range(len(tokens) - k + 1):
#                 ngrams.add(tuple(tokens[i:i+k]))
#         return ngrams

#     for i in range(n):
#         for j in range(i, n):
#             if i == j:
#                 A[i, j] = 1.0
#                 continue
#             ngrams_i = get_ngrams(agent_responses[agent_names[i]])
#             ngrams_j = get_ngrams(agent_responses[agent_names[j]])
#             if not ngrams_i and not ngrams_j:
#                 sim = 1.0
#             else:
#                 sim = len(ngrams_i & ngrams_j) / len(ngrams_i | ngrams_j)
#             A[i, j] = A[j, i] = sim
#     return A

def compute_text_adjacency_matrix(agent_names, agent_responses):
    n = len(agent_names)
    A = np.zeros((n, n))

    def get_ngrams(text, ngram=3):
        tokens = str(text).lower().split()
        ngrams = set()
        for k in range(1, ngram + 1):
            for i in range(len(tokens) - k + 1):
                ngrams.add(tuple(tokens[i:i+k]))
        return ngrams

    # 🔥 CACHE ONCE
    cached = {
        name: get_ngrams(agent_responses[name])
        for name in agent_names
    }

    for i in range(n):
        n_i = cached[agent_names[i]]
        for j in range(i, n):
            if i == j:
                A[i, j] = 1.0
                continue

            n_j = cached[agent_names[j]]

            if not n_i and not n_j:
                sim = 1.0
            else:
                sim = len(n_i & n_j) / len(n_i | n_j)

            A[i, j] = A[j, i] = sim

    return A

def graph_cut(A, names):
    """Spectral 2-way cut đơn giản (dùng Laplacian)"""
    # Tính Laplacian
    D = np.diag(np.sum(A, axis=1))
    L = D - A

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]   # vector thứ 2 (Fiedler vector)

    # Phân nhóm theo dấu của Fiedler vector
    group1 = [names[i] for i in range(len(names)) if fiedler_vector[i] >= 0]
    group2 = [names[i] for i in range(len(names)) if fiedler_vector[i] < 0]

    return {'groups': {'group_1': group1, 'group_2': group2}}


def goodness_of_cut_func(A, group_indices, n_group_indices, method='conductance'):
    cut = float(np.sum(A[np.ix_(group_indices, n_group_indices)]))
    if method == 'conductance':
        vol_s = float(np.sum(A[group_indices, :]) - len(group_indices))
        vol_sbar = float(np.sum(A[n_group_indices, :]) - len(n_group_indices))
        return cut / (min(vol_s, vol_sbar) + 1e-10)
    elif method == 'cutratio':
        return cut / (np.sum(A) - A.shape[0])
    else:  # ngc hoặc mặc định
        vol_s = float(np.sum(A[group_indices, :]) - len(group_indices))
        vol_sbar = float(np.sum(A[n_group_indices, :]) - len(n_group_indices))
        return (cut / vol_s) + (cut / vol_sbar)


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

    accuracy = {}
    
    # print('--- Starting evaluation of generations ---')
    
    for i, gen in enumerate(tqdm(generations, desc="Processing generations")):
        # print(f"\n--- Evaluating sample {i+1}/{len(generations)} ---")
        # --- Find the least uncertain samples ---
        cleaned_texts = gen["cleaned_generated_texts"]
        extracted_answers = gen["extracted_answers"] if "extracted_answers" in gen else [None] * len(cleaned_texts)
        samples_avg_nll = gen["samples_avg_nll"]
        samples_nll = gen["samples_nll"]

        selected_index = modex_select(cleaned_texts, adjacency='text', tau=0.8, goodness_of_cut='conductance', emb_encoder=embed_model)

        # print(f"Selected index by ModeX: {selected_index}")
        
        # --- Ranking and find samples ---
        if len(samples_nll) == 0:
            modex_sample = None

        else:
            if args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med']:

                modex_sample = extracted_answers[selected_index]
            
            else:
            
                modex_sample = cleaned_texts[selected_index]
    
        # --- Evaluation ---        
        for method, sample in zip(['modex'], [modex_sample]):
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
        
        # print(f"Processed sample {i+1}/{len(generations)}")
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
        "seed": args.seed,
        "modex": args.modex
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
    parser.add_argument('--modex', action='store_true', help='Whether to compute modex')
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0, help='Fraction of data to use for evaluation (for quick testing)')
    parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for binary classification of correctness (used for non-math datasets)')
    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
    parser.add_argument('--eval_method', type=str, default='rougeL', help='Evaluation method for non-math datasets (e.g., rougeL or api)')
    parser.add_argument('--api_type', type=str, default='cohere', choices=['gemini', 'cohere'], help='API type for LLM evaluation')

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    args.timestamp = timestamp

    print(f"RANKING: Dataset={args.dataset}, Model={args.model}, N={args.n_samples}, F={args.fraction_of_data_to_use}, T={args.threshold}, S={args.seed}, E={args.eval_method}, A={args.api_type}.")
    set_seed(args.seed)
    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    print(f"Total evaluation time: {end_time - start_time}")