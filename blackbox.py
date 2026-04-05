from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
import cohere
import api_key
import os
import re

def load_mmlu_pro(split='test', n_sample_per_cat=10):
    # split = 'validation' | 'test'
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")[split]
    dataset = pd.DataFrame(dataset)
    
    # Filter
    selected_categories = dataset['category'].unique()
    selected_question_idx = []
    for cat in selected_categories:
        question_idx = dataset.loc[dataset['category'] == cat, 'question_id'].tolist()
        selected_question_idx.extend(question_idx[:n_sample_per_cat])
    
    dataset = dataset[dataset['question_id'].isin(selected_question_idx)]

    # Prepare questions and labels
    questions, labels = [], []
    choices = "ABCDEFGHIJ"
    
    template = '{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n(E) {}\n(F) {}\n(G) {}\n(H) {}\n(I) {}\n(J) {}\n\n'
    for query, options, answer in zip(dataset['question'], dataset['options'], dataset['answer_index']):
        if len(options) != 10 :
            continue
        question = template.format(query, options[0], options[1], options[2], options[3], options[4], options[5], 
                                   options[6], options[7], options[8], options[9])
        label = f"({choices[int(answer)]})"
        questions.append(question)
        labels.append(label)

    return questions, labels


def extract_math_response(text, data_type='math'):
    try:
        pred = re.findall(r"\{(.*?)\}", text)[-1]
        pred = pred.replace("final answer:", "").strip()
        
        if data_type == 'math':
            text = int(pred)
        elif data_type == 'mc':
            
            if len(pred) == 0:
                text = ""
            elif len(pred) < 3:
                pred = pred[0]
                text = f"({pred})"
            else:
                pred = pred[1]
                text = f"({pred})"
            
        else:
            raise ValueError("Unknown data type")

    except :
        text =  ""
        
    return text


def main(args):
    if args.client == 'cohere':
        client = cohere.ClientV2(api_key=api_key.cohere_api_key)
        model = api_key.cohere_model
    else:
        raise ValueError("Unsupported client")

    SUFFIX_MATH = " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 12.34}'. Let's think step by step."
    SUFFIX_MC = " Make sure to state your final answer choice in curly brackets at the very end of your response, just like: '{final answer: (A)}'. Let's think step by step."

    if args.dataset == 'aime25':
        ds = load_dataset("MathArena/aime_2025")['train']
        SUFFIX = SUFFIX_MATH
        data_type = 'math'
    elif args.dataset == 'mmlu_pro':
        questions, labels = load_mmlu_pro(split='test')
        SUFFIX = SUFFIX_MC
        data_type = 'mc'
    else:
        raise ValueError("Unsupported dataset")

    results = {}

    for idx in tqdm(range(len(questions))):
        sample_question = questions[idx]

        message = [
            {"role": "user", "content": sample_question + SUFFIX},
        ]

        list_res = []
        for i in range(args.n_samples):
            res = client.chat(messages=message, model=model, temperature=args.temperature, p=args.top_p, max_tokens=args.max_tokens)
            list_res.append(res.message.content[0].text)
            
        results[idx] = list_res
        
    # Save
    rows = []

    for idx, answers in results.items():
        row = {"idx": idx}
        for i, ans in enumerate(answers):
            pred = extract_math_response(ans, data_type=data_type)
            
            row[f"gen_{i}"] = ans
            row[f"pred_{i}"] = pred
        rows.append(row)

    df = pd.DataFrame(rows)

    # save
    folder_path = f"/home/s224852302/RDS/results/"
    os.makedirs(folder_path, exist_ok=True)
    df.to_csv(f"{folder_path}{args.client}_{args.dataset}_{args.n_samples}_{args.temperature}_{args.top_p}_{args.max_tokens}.csv", index=False)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=str, default="cohere", help="The LLM client to use (e.g., cohere)")
    parser.add_argument("--dataset", type=str, default="aime25", help="The dataset to evaluate on (e.g., aime25)")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to generate for each question")
    parser.add_argument("--temperature", type=float, default=1, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.99, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")

    args = parser.parse_args()
    main(args)