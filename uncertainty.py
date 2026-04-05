import warnings
warnings.filterwarnings('ignore')

import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr
import config
import json
import argparse
from collections import defaultdict
from collections import Counter
import numpy as np
from datetime import datetime

import logging
logging.basicConfig(level=logging.ERROR)

import os
import pandas as pd

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import (
    calculate_p_true,
    semantic_volume,
    compute_graph_baselines,
    load_huggingface_model,
    set_seed,
    compute_weighted_mean,
    compute_semantic_entropy, 
    compute_deg_semantic_density, 
    pro_score, 
    compute_eigen_embed,
    compute_ece,
    minmax_normalize
    )


#### -------------------- SCORING --------------------
def main(args, semantic_model, semantic_tokenizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer(args.embed_model).to(device)

    # --- Load generations ---
    gen_path = f"{config.output_dir}/{args.dataset}_{args.model}_N={args.n_samples}_F={args.fraction_of_data_to_use}_A={args.api_type}_S={args.seed}__generation.pkl"
    with open(gen_path, "rb") as infile:
        generations = pickle.load(infile)

    labels = []
    norm_dict = defaultdict(list)
    
    if args.p_true:
        tokenizer, model = load_huggingface_model(model_name=args.model)

    for i, gen in enumerate(tqdm(generations, desc="Processing generations")):
        
        with torch.no_grad():
            # --- Label ---
            if args.dataset in ['gsm8k', 'svamp', 'arith', 'formal_logic', 'arith_long']:
                label = 1 - int(gen['eval_score'] == 1.0)
            else:
                if args.eval_method == 'llm':
                    label = 1 - int(gen['llm_label'] == 1.0)
                elif args.eval_method == 'rougeL':
                    label = 1 - int(gen['eval_score'] > args.threshold)
                else:
                    raise ValueError(f"Unsupported eval method: {args.eval_method}")
            labels.append(label)

            extracted_answers = gen.get("extracted_answers", [])
            cleaned_texts = gen["cleaned_generated_texts"]
            samples_avg_nll = gen["samples_avg_nll"]
            samples_nll = gen["samples_nll"]

            # --- Embeddings ---
            if args.dataset in ['gsm8k', 'formal_logic', 'arith_long']:
                rds_texts = [str(j) for j in extracted_answers]
                gen["cleaned_generated_texts"] = rds_texts # at final answers level not full reasoning path
            else:
                rds_texts = cleaned_texts
                            
            embeddings = embed_model.encode(rds_texts, convert_to_tensor=True, device=device)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # --- RDS base version ---
            mean_embedding = torch.mean(embeddings, dim=0)
            diffs = embeddings - mean_embedding 
            rds_scores = torch.norm(diffs, p=1, dim=1)
            rds = rds_scores.sum().item()
            rds_l2 = torch.norm(diffs, p=2, dim=1).sum().item()
            
            # --- RDS weighted version ---
            probs = np.exp(-np.array(samples_avg_nll))
            probs /= probs.sum()
            probs = torch.tensor(probs, dtype=torch.float32, device=device)
            weighted_mean_embeddings = compute_weighted_mean(embeddings, probs)        
            diffs_weighted = embeddings - weighted_mean_embeddings.unsqueeze(0)
            weighted_rds_scores = torch.norm(diffs_weighted, p=1, dim=1)
            weighted_rds = (probs * weighted_rds_scores).sum().item()
            weighted_rds_l2 = (probs * torch.norm(diffs_weighted, p=2, dim=1)).sum().item()
            
            # --- EigenEmbed ---
            eigen_embed = compute_eigen_embed(embeddings, alpha=1e-3)
            
            # --- Store norms ---
            norm_dict["ANLL"].append(float(np.mean(samples_avg_nll)))
            norm_dict["NLL"].append(float(np.mean(samples_nll)))
            norm_dict["EigenEmbed"].append(eigen_embed)
            norm_dict["RDS Score (base)"].append(rds)
            norm_dict["RDS L2 (base)"].append(rds_l2)
            norm_dict["RDS Score (weighted)"].append(weighted_rds)
            norm_dict["RDS L2 (weighted)"].append(weighted_rds_l2)
            ### PRO
            norm_dict["PRO"].append(pro_score(gen))
            
            ### Semantic Baselines
            if args.semantic_baselines:
                # Semantic Entropy
                sem_entropy, dse, semantic_sets = compute_semantic_entropy(gen, semantic_model, semantic_tokenizer)
                norm_dict["SE"].append(sem_entropy)
                norm_dict["DSE"].append(dse)
                norm_dict["NumSet"].append(semantic_sets)

                # Semantic Density
                deg_old, sd_val = compute_deg_semantic_density(gen, semantic_model, semantic_tokenizer)
                norm_dict["SD"].append(sd_val)
                norm_dict["Deg_Old"].append(deg_old)
                
                # LexicalSim / Deg / Ecc / EigenLap / KLE
                graph_baselines = compute_graph_baselines(cleaned_texts, semantic_model=semantic_model, semantic_tokenizer=semantic_tokenizer, device=device)
                for method, score in graph_baselines.items():
                    norm_dict[method].append(score)
                    
                sv = semantic_volume(embeddings)
                norm_dict["SemanticVolume"].append(sv)
                
            if args.p_true:           
                p_true = calculate_p_true(
                    model=model,
                    tokenizer=tokenizer,
                    question=gen["question"],
                    most_probable_answer=str(gen["greedy_text"]),
                    brainstormed_answers=cleaned_texts,
                    device=device
                    )
                norm_dict["P(True)"].append(p_true)       
                
            ### Self-Consistency
            if args.dataset in ['gsm8k', 'formal_logic', 'arith_long']:
                freq = Counter(extracted_answers)
            else:
                freq = Counter(cleaned_texts)
            major_sample_count = freq.most_common(1)[0][1]
            major_score = major_sample_count / len(cleaned_texts)
            norm_dict['SC'].append(1 - major_score)

    # --- AUROC reporting ---
    results = {}
    correlation_results = {}

    for method, values in norm_dict.items():
        values = [
            v.detach().cpu().item() if torch.is_tensor(v) else v
            for v in values
        ]
        values = np.array(values)

        auc = roc_auc_score(labels, values)
        results[f"AUC_{method}"] = round(auc, 4)

        sp, _ = spearmanr(labels, values)
        pr, _ = pearsonr(labels, values)

        correlation_results[f"Spearman_{method}"] = round(sp, 4)
        correlation_results[f"Pearson_{method}"] = round(pr, 4)
        
    ece_results = {}
    for method, values in norm_dict.items():
        values = [
            v.detach().cpu().item() if torch.is_tensor(v) else v
            for v in values
        ]
        values = np.array(values)

        # uncertainty → confidence
        norm_unc = minmax_normalize(values)
        confidence = 1.0 - norm_unc

        ece = compute_ece(confidence, np.array(labels))
        ece_results[f"ECE_{method}"] = round(ece, 4)
    
    print("\n=== Metric Performance (ROC-AUC) ===")
    for method, auc in results.items():
        print(f"{method:45s} → ROC-AUC: {auc:.4f}")
    print("====================================\n")

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
        "seed": args.seed,
        "fraction_of_data_to_use": args.fraction_of_data_to_use,
    }
    row.update(results) # AUC: discriminative power
    row.update(correlation_results) # ranking (Spearman)
    row.update(ece_results) # ECE: calibration
    new_row_df = pd.DataFrame([row])

    # --- Check if file exists ---
    tsv_file = f'results/uncertainty_logs.tsv'
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
    parser = argparse.ArgumentParser(description='Compute uncertainty scores for generated sequences')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--embed_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--semantic_baselines', action='store_true', help='Whether to compute semantic baselines (SE, Deg, SD)')
    parser.add_argument('--p_true', action='store_true', help='Whether to compute P(True) for each sample')
    parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for binary classification of correctness (used for non-math datasets)')
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0, help='Fraction of data to use for evaluation (for quick testing)')
    parser.add_argument('--eval_method', type=str, default='rougeL', help='Evaluation method for non-math datasets (e.g., rougeL or api)')
    parser.add_argument('--api_type', type=str, default='cohere', choices=['gemini', 'cohere'], help='API type for LLM evaluation')
    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    args.timestamp = timestamp
    
    set_seed(args.seed)
    
    # --- Load semantic model and tokenizer ---
    semantic_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    semantic_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to('cuda')
    semantic_model.eval()
    
    print(f"UNCERTAINTY: Dataset={args.dataset}, Model={args.model}, EB={args.embed_model}, N={args.n_samples}, F={args.fraction_of_data_to_use}, T={args.threshold}, S={args.seed}, E={args.eval_method}, A={args.api_type}.")
    start_time = datetime.now()
    main(args, semantic_model, semantic_tokenizer)
    end_time = datetime.now()
    print(f"Total scoring time: {end_time - start_time}")