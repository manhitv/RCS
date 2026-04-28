import os
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
import cohere
import re

from . import api_key
from .utils import extract_code_response

def load_hle(split='test', max_samples=100):
    # split = 'validation' | 'test'
    dataset = load_dataset("cais/hle")[split]
    dataset = pd.DataFrame(dataset)
    
    # Filter
    # 1. remove multimodal
    df = dataset[dataset["image"] == '']

    # 3. short answer / MCQ only
    df = df[df["answer"].apply(lambda x: len(x) <= 5)] # 1464 samples
    
    print(f"Loaded HLE dataset with {len(df)} samples after filtering.")

    # Prepare questions and labels
    questions, labels = [], []
    for question, answer in zip(df['question'], df['answer']) :
        questions.append(question)
        labels.append(answer)

    return questions[:max_samples], labels[:max_samples]

def load_gpqa(split='validation'):
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")['train']
    dataset = pd.DataFrame(dataset)
    questions, labels = [], []
    template = '{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n\n'
    for ctx, A, B, C, D in zip(dataset['Question'], dataset['Correct Answer'], dataset['Incorrect Answer 1'], dataset['Incorrect Answer 2'], dataset['Incorrect Answer 3']):
        
        question = template.format(ctx, A, B, C, D)
        label = f"(A)"
        questions.append(question)
        labels.append(label)
    
    return questions, labels


def load_cruxeval(max_samples=100):
    ds = load_dataset('cruxeval-org/cruxeval')['test']
    format_code = "Code: {code}\n\nInput: {input}"
    questions = [format_code.format(code=item['code'], input=item['input']) for item in ds]
    answers = [item['output'] for item in ds]
    
    return questions[:max_samples], answers[:max_samples]


def load_bigbenchhard_nav(split='validation'):
    dataset = load_dataset("maveriq/bigbenchhard", "navigate")['train']
    dataset = pd.DataFrame(dataset)
    questions, labels = [], []
    for q, a in zip(dataset['input'], dataset['target']):
        questions.append(q)
        labels.append(a)

    return questions, labels

def load_bigbenchhard_date(split='validation'):
    dataset = load_dataset("maveriq/bigbenchhard", "date_understanding")['train']
    dataset = pd.DataFrame(dataset)
    questions, labels = [], []
    for q, a in zip(dataset['input'], dataset['target']):
        questions.append(q)
        labels.append(a)

    return questions, labels


def extract_answer_response(text, data_type='math'):
    try:
        pred = re.findall(r"\{(.*?)\}", text)[-1]
        pred = pred.replace("final answer:", "").strip()
        
        if data_type == 'math':
            text = int(pred)
        elif data_type in ['mc', 'bbh_date']:
            
            if len(pred) == 0:
                text = ""
            elif len(pred) < 3:
                pred = pred[0]
                text = f"({pred})"
            else:
                pred = pred[1]
                text = f"({pred})"
        elif data_type in ['hle', 'bbh_nav']:
            text = pred
            
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
    SUFFIX_HLE = " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: Z+Z+Z+Z+Z}'. Let's think step by step."
    # SUFFIX_GPQA = " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: Z+Z+Z+Z+Z}'. Let's think step by step."
    SUFFIX_BBH_DATE = "Do not include any reasoning steps. Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: (A)}'."
    SUFFIX_BBH_NAV = "Do not include any reasoning steps. Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: Yes}'."
    SUFFIX_CODE = """You are given a Python function and some inputs. 
Your task is to determine the exact output of the function when called with those inputs.

Think step by step inside your mind, but **DO NOT output any reasoning, explanation, or extra text**.
Your response MUST be **ONLY** a valid JSON object in this exact format:
{"answer": <final_output>}
Do not include any markdown, code blocks, or additional text outside the JSON."""
    
    if args.dataset == 'aime25':
        ds = load_dataset("MathArena/aime_2025")['train']
        SUFFIX = SUFFIX_MATH
        data_type = 'math'
    elif args.dataset == 'mmlu_pro':
        questions, labels = load_mmlu_pro(split='test')
        SUFFIX = SUFFIX_MC
        data_type = 'mc'
    elif args.dataset == 'hle':
        questions, labels = load_hle(split='test')
        SUFFIX = SUFFIX_HLE
        data_type = 'hle'
    elif args.dataset == 'gpqa':
        questions, labels = load_gpqa(split='train')
        SUFFIX = SUFFIX_MC
        data_type = 'mc'
    elif args.dataset == 'bbh_nav':
        questions, labels = load_bigbenchhard_nav(split='train')
        SUFFIX = SUFFIX_BBH_NAV
        data_type = 'bbh_nav'
    elif args.dataset == 'bbh_date':
        questions, labels = load_bigbenchhard_date(split='train')
        SUFFIX = SUFFIX_BBH_DATE
        data_type = 'bbh_date'

    elif args.dataset == 'cruxeval':
        questions, labels = load_cruxeval()
        SUFFIX = SUFFIX_CODE
        data_type = 'code'

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
            if data_type == 'code':
                pred = extract_code_response(ans)
            else:
                pred = extract_answer_response(ans, data_type=data_type)
            
            row[f"gen_{i}"] = ans
            row[f"pred_{i}"] = pred
        rows.append(row)

    df = pd.DataFrame(rows)

    # save
    folder_path = f"/home/s224852302/RCS/results/"
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