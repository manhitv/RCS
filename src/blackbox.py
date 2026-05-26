"""Black-box (API-only) sampling pipeline. Helpers live in `utils.py`."""

import os
import argparse

import cohere
import pandas as pd
from tqdm import tqdm

from . import api_key
from .utils import (
    extract_code_response,
    extract_answer_response,
    get_blackbox_dataset,
)


def main(args):
    if args.client == "cohere":
        client = cohere.ClientV2(api_key=api_key.cohere_api_key)
        model = api_key.cohere_model
    else:
        raise ValueError("Unsupported client")

    questions, _labels, suffix, data_type = get_blackbox_dataset(args.dataset)

    results = {}
    for idx in tqdm(range(len(questions))):
        message = [{"role": "user", "content": questions[idx] + suffix}]

        list_res = []
        for _ in range(args.n_samples):
            res = client.chat(
                messages=message, model=model,
                temperature=args.temperature, p=args.top_p, max_tokens=args.max_tokens,
            )
            list_res.append(res.message.content[0].text)
        results[idx] = list_res

    rows = []
    for idx, answers in results.items():
        row = {"idx": idx}
        for i, ans in enumerate(answers):
            pred = extract_code_response(ans) if data_type == "code" \
                else extract_answer_response(ans, data_type=data_type)
            row[f"gen_{i}"] = ans
            row[f"pred_{i}"] = pred
        rows.append(row)

    folder_path = "results/"
    os.makedirs(folder_path, exist_ok=True)
    out_path = (
        f"{folder_path}{args.client}_{args.dataset}_{args.n_samples}_"
        f"{args.temperature}_{args.top_p}_{args.max_tokens}.csv"
    )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=str, default="cohere", help="LLM client (e.g., cohere)")
    parser.add_argument("--dataset", type=str, default="aime25", help="Black-box benchmark name")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()
    main(args)
