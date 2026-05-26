"""Compute Best-of-N accuracy across every ranking metric in one pass.

Metrics computed for every run
  - NLL / avg-NLL                          (likelihood-based)
  - RDS_base / RDS_freq / RDS_prob         (Fréchet-mean RDS)
  - RDS_medoid + freq/prob variants        (medoid-based RDS)
  - SCW  / RDS_cosine                      (cosine-similarity-weighted)
  - majority vote

Optional metrics (CLI flags)
  - --self_certainty   self-certainty + power-vote
  - --modex            ModeX (recursive spectral graph cut)
  - --include_oracle   oracle upper bound
  - --raw_answers      RDS variants on raw numeric answers (math only)
  - --full_answers     embed full reasoning traces instead of extracted answers
"""

import os
import argparse
import pickle
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import evaluate
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from . import config
from .utils import (
    MODEL_PATH_DICT,
    EXTRACTED_ANSWER_DATASETS,
    set_seed,
    evaluation_sample,
    compute_weighted_mean,
    compute_rds,
    compute_rds_raw,
    compute_rds_raw_medoid_weighted,
    compute_medoid,
    compute_self_certainty_scores,
    get_self_certainty_sample,
    modex_select,
)

warnings.filterwarnings("ignore")


def main(args):
    experiment_id = os.getpid()
    cache_dir = f"/tmp/rouge_cache_{experiment_id}"
    os.environ["HF_EVALUATE_CACHE"] = cache_dir
    rouge = evaluate.load("rouge", experiment_id=experiment_id, cache_dir=cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(args.embed_model).to(device)

    # --- Load generations ---
    gen_path = (
        f"{config.output_dir}/{args.dataset}_{args.model}"
        f"_N={args.n_samples}_F={args.fraction_of_data_to_use}"
        f"_A={args.api_type}_S={args.seed}__generation.pkl"
    )
    with open(gen_path, "rb") as infile:
        generations = pickle.load(infile)

    # Pre-compute self-certainty for the whole dataset once (heavy model load).
    if args.self_certainty:
        sc_cache_path = (
            f"{config.output_dir}/{args.dataset}_{args.model}"
            f"_N={args.n_samples}_F={args.fraction_of_data_to_use}"
            f"_A={args.api_type}_S={args.seed}__self_certainty.pkl"
        )
        os.makedirs(os.path.dirname(sc_cache_path), exist_ok=True)

        if os.path.exists(sc_cache_path):
            with open(sc_cache_path, "rb") as f:
                all_self_certainty = pickle.load(f)
        else:
            prompts = [g["prompt"] for g in generations]
            generated_texts_list = [g["cleaned_generated_texts"] for g in generations]
            all_self_certainty = compute_self_certainty_scores(
                model_dir=MODEL_PATH_DICT[args.model],
                prompts=prompts,
                generated_texts_list=generated_texts_list,
                batch_size=4,
                device=device,
            )
            with open(sc_cache_path, "wb") as f:
                pickle.dump(all_self_certainty, f)
            print(f"Saved self-certainty scores to {sc_cache_path}")

        for i, gen in enumerate(generations):
            gen["samples_ce"] = all_self_certainty[i]

    accuracy = {"greedy": []}
    use_extracted = args.dataset in EXTRACTED_ANSWER_DATASETS

    for i, gen in enumerate(tqdm(generations, desc="Processing generations")):
        cleaned_texts = gen["cleaned_generated_texts"]
        extracted_answers = gen.get("extracted_answers", [None] * len(cleaned_texts))
        samples_avg_nll = gen["samples_avg_nll"]
        samples_nll = gen["samples_nll"]

        blank_indices = []
        if args.ignore_null:
            blank_indices = [
                idx for idx, ans in enumerate(extracted_answers) if ans in [None, ""]
            ]
            cleaned_texts = [t for idx, t in enumerate(cleaned_texts) if idx not in blank_indices]
            extracted_answers = [a for idx, a in enumerate(extracted_answers) if idx not in blank_indices]
            samples_avg_nll = [n for idx, n in enumerate(samples_avg_nll) if idx not in blank_indices]
            samples_nll = [n for idx, n in enumerate(samples_nll) if idx not in blank_indices]

        # --- Choose text source for embeddings ---
        if use_extracted:
            texts_for_embedding = cleaned_texts if args.full_answers \
                else [str(j) for j in extracted_answers]
        else:
            texts_for_embedding = cleaned_texts

        embeddings = embed_model.encode(
            texts_for_embedding, convert_to_tensor=True, device=device
        )
        candidate_pool = extracted_answers if use_extracted else cleaned_texts

        # ============== 1. RDS — Fréchet-mean centers ==============
        # 1a. Uniform
        rds_base_center = torch.mean(embeddings, dim=0)
        rds_base = compute_rds(embeddings, rds_base_center)

        # 1b. Frequency-weighted
        freq_counts = Counter(texts_for_embedding)
        probs_freq = np.array(
            [freq_counts[t] for t in texts_for_embedding], dtype=np.float32
        )
        probs_freq /= probs_freq.sum()
        freq_t = torch.tensor(probs_freq, dtype=torch.float32, device=device)
        rds_freq_center = compute_weighted_mean(embeddings, freq_t)
        rds_freq = compute_rds(embeddings, rds_freq_center)

        # 1c. Probability-weighted
        probs_prob = np.exp(-np.array(samples_avg_nll))
        probs_prob /= probs_prob.sum()
        prob_t = torch.tensor(probs_prob, dtype=torch.float32, device=device)
        rds_prob_center = compute_weighted_mean(embeddings, prob_t)
        rds_prob = compute_rds(embeddings, rds_prob_center)

        # ============== 2. RDS — medoid centers ==============
        rds_medoid = compute_rds(embeddings, compute_medoid(embeddings))
        rds_medoid_freq = compute_rds(embeddings, compute_medoid(embeddings, weights=freq_t))
        rds_medoid_prob = compute_rds(embeddings, compute_medoid(embeddings, weights=prob_t))

        # ============== 3. RDS — raw numeric answers (math only) ==============
        compute_raw = args.dataset in ["gsm8k", "arith_long"] and args.raw_answers
        if compute_raw:
            rds_raw_base = compute_rds_raw(extracted_answers)
            rds_raw_freq = compute_rds_raw(extracted_answers, weights=probs_freq)
            rds_raw_prob = compute_rds_raw(extracted_answers, weights=probs_prob)
            rds_raw_medoid = compute_rds_raw_medoid_weighted(extracted_answers)
            rds_raw_medoid_freq = compute_rds_raw_medoid_weighted(extracted_answers, weights=probs_freq)
            rds_raw_medoid_prob = compute_rds_raw_medoid_weighted(extracted_answers, weights=probs_prob)

        # ============== 4. Cosine-similarity-weighted (SCW + RDS_cosine) ==============
        # SCW = discrete vote weighted by cosine-sim sums (majority's cosine cousin).
        # RDS_cosine = Fréchet-mean RDS with the same sim_sum as continuous weights.
        full_text_embeddings = embed_model.encode(
            cleaned_texts, convert_to_tensor=True, device=device
        )
        norms = torch.norm(full_text_embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        normed = full_text_embeddings / norms
        sim_sum = torch.mm(normed, normed.t()).sum(dim=1)  # (N,)

        answer_to_indices = {}
        for idx, text in enumerate(cleaned_texts):
            answer_to_indices.setdefault(text, []).append(idx)
        answer_group_scores = {
            text: sum(sim_sum[idx].item() for idx in indices)
            for text, indices in answer_to_indices.items()
        }
        scw_best_text = max(answer_group_scores, key=answer_group_scores.get)
        scw_idx = max(answer_to_indices[scw_best_text], key=lambda idx: sim_sum[idx].item())

        probs_cosine = sim_sum / sim_sum.sum().clamp(min=1e-8)
        rds_cosine_center = compute_weighted_mean(embeddings, probs_cosine)
        rds_cosine = compute_rds(embeddings, rds_cosine_center)

        # ============== 5. Pick samples by argmin / vote ==============
        if len(samples_nll) == 0:
            method_samples = {k: None for k in [
                "nll", "avg_nll", "rds_base", "rds_freq", "rds_prob",
                "rds_medoid", "rds_medoid_freq", "rds_medoid_prob",
                "scw", "rds_cosine", "majority",
            ]}
            if compute_raw:
                for k in ["rds_raw_base", "rds_raw_freq", "rds_raw_prob",
                          "rds_raw_medoid", "rds_raw_medoid_freq", "rds_raw_medoid_prob"]:
                    method_samples[k] = None
            if args.include_oracle:
                method_samples["oracle"] = None
        else:
            def pick_t(tensor_scores):
                return candidate_pool[torch.argmin(tensor_scores).item()]

            def pick_np(np_scores):
                return candidate_pool[int(np.argmin(np_scores))]

            method_samples = {
                "nll": candidate_pool[int(np.argmin(samples_nll))],
                "avg_nll": candidate_pool[int(np.argmin(samples_avg_nll))],
                "rds_base": pick_t(rds_base),
                "rds_freq": pick_t(rds_freq),
                "rds_prob": pick_t(rds_prob),
                "rds_medoid": pick_t(rds_medoid),
                "rds_medoid_freq": pick_t(rds_medoid_freq),
                "rds_medoid_prob": pick_t(rds_medoid_prob),
                "scw": candidate_pool[scw_idx],
                "rds_cosine": pick_t(rds_cosine),
                "majority": Counter(candidate_pool).most_common(1)[0][0],
            }

            if compute_raw:
                method_samples["rds_raw_base"] = pick_np(rds_raw_base)
                method_samples["rds_raw_freq"] = pick_np(rds_raw_freq)
                method_samples["rds_raw_prob"] = pick_np(rds_raw_prob)
                method_samples["rds_raw_medoid"] = pick_np(rds_raw_medoid)
                method_samples["rds_raw_medoid_freq"] = pick_np(rds_raw_medoid_freq)
                method_samples["rds_raw_medoid_prob"] = pick_np(rds_raw_medoid_prob)

            if args.include_oracle:
                oracle_sample = "None"
                for ans in candidate_pool:
                    acc = evaluation_sample(
                        dataset=args.dataset, text=ans, answer=gen["answer"],
                        question=gen.get("question"), rouge=rouge,
                        api_type=args.api_type, eval_method=args.eval_method,
                        threshold=args.threshold,
                    )
                    if acc == 1:
                        oracle_sample = ans
                        break
                method_samples["oracle"] = oracle_sample

        # --- Self-certainty (uses pre-computed cache) ---
        if args.self_certainty:
            if "samples_ce" in gen:
                sc_scores = np.array(gen["samples_ce"])
                if args.ignore_null:
                    sc_scores = [s for idx, s in enumerate(sc_scores) if idx not in blank_indices]
                method_samples["self_certainty"] = (
                    None if len(sc_scores) == 0
                    else get_self_certainty_sample(sc_scores, candidate_pool)
                )
            else:
                method_samples["self_certainty"] = None

        # --- ModeX ---
        if args.modex:
            modex_idx = modex_select(
                cleaned_texts, adjacency="text", tau=0.8,
                goodness_of_cut="conductance", emb_encoder=embed_model,
            )
            method_samples["modex"] = candidate_pool[modex_idx]

        # --- Evaluate every method ---
        for method, sample in method_samples.items():
            acc = evaluation_sample(
                dataset=args.dataset, text=sample, answer=gen["answer"],
                question=gen.get("question"), rouge=rouge,
                api_type=args.api_type, eval_method=args.eval_method,
                threshold=args.threshold,
            )
            accuracy.setdefault(method, []).append(acc)
            if i < 3:
                print(f"Sample {i} | Method: {method} | Acc: {acc} | Sample: {sample}...")

        # Greedy baseline
        greedy_acc = evaluation_sample(
            dataset=args.dataset, text=gen["greedy_text"], answer=gen["answer"],
            question=gen.get("question"), rouge=rouge,
            api_type=args.api_type, eval_method=args.eval_method,
            threshold=args.threshold,
        )
        accuracy["greedy"].append(greedy_acc)

    # --- Reporting ---
    results = {}
    print("\n=== Metric Performance ===")
    for method, values in accuracy.items():
        final_acc = float(np.mean(values))
        results[method] = round(final_acc, 4)
        print(f"{method:55s} → ACC: {final_acc:.4f}")

    # --- Append to TSV ---
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

    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    tsv_file = f"{result_dir}/ranking_logs.tsv"

    if os.path.exists(tsv_file):
        df = pd.read_csv(tsv_file, sep="\t")
        all_cols = sorted(set(df.columns).union(new_row_df.columns))
        df = df.reindex(columns=all_cols)
        new_row_df = new_row_df.reindex(columns=all_cols)
        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        df = new_row_df

    df.to_csv(tsv_file, sep="\t", index=False)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Best-of-N accuracy for every ranking metric in one pass."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--self_certainty", action="store_true",
                        help="Compute self-certainty + power-vote baseline")
    parser.add_argument("--modex", action="store_true",
                        help="Compute ModeX (spectral graph-cut) baseline")
    parser.add_argument("--fraction_of_data_to_use", type=float, default=1.0,
                        help="Fraction of data to use (for quick testing)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Correctness threshold for short-form QA (non-math)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_method", type=str, default="rougeL",
                        help="Evaluation method for non-math datasets (rougeL | llm_eval)")
    parser.add_argument("--api_type", type=str, default="cohere",
                        choices=["gemini", "cohere"],
                        help="API type for LLM-based evaluation")
    parser.add_argument("--ignore_null", action="store_true",
                        help="Drop samples whose extracted answer is null/empty")
    parser.add_argument("--raw_answers", action="store_true",
                        help="Compute RDS variants on raw numeric answers (math only)")
    parser.add_argument("--full_answers", action="store_true",
                        help="Embed full reasoning traces instead of extracted answers (math only)")
    parser.add_argument("--include_oracle", action="store_true",
                        help="Include oracle upper bound (slow for non-math datasets)")
    args = parser.parse_args()

    args.timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    print(
        f"RANKING: Dataset={args.dataset}, Model={args.model}, N={args.n_samples}, "
        f"F={args.fraction_of_data_to_use}, T={args.threshold}, S={args.seed}, "
        f"E={args.eval_method}, A={args.api_type}, I={args.ignore_null}."
    )
    set_seed(args.seed)
    start_time = datetime.now()
    main(args)
    print(f"Total evaluation time: {datetime.now() - start_time}")
