"""Utility functions for RCS: dataset loading, generation, metrics, and baselines.

Sections
--------
1.  Imports & constants
2.  Environment / seeding
3.  Model loading
4.  Text cleaning & answer extraction
5.  Instruction templates
6.  Dataset loaders (open-weight + black-box benchmarks)
7.  Generation + NLL
8.  LLM-based evaluation
9.  Calibration helpers
10. Distance / RDS metrics
11. Self-certainty
12. ModeX
13. Black-box prompt suffixes
"""

import os
import re
import ast
import json
import random
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import SamplingParams

import cohere
from google import genai

from . import config, api_key

logging.basicConfig(level=logging.ERROR)


# =====================================================================
# 1. CONSTANTS
# =====================================================================
ANSWER_TRIGGER = "The answer is"
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

MODEL_PATH_DICT = {
    "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "falcon3-1b": "tiiuae/falcon3-1b-instruct",
    "falcon3-7b": "tiiuae/falcon3-7b-instruct",
    "falcon3-10b": "tiiuae/falcon3-10b-instruct",
    "gemma-7b": "google/gemma-7b-it",
    "gemma-2b": "google/gemma-2b-it",
    "gemma2-2b": "google/gemma-2-2b-it",
    "gemma2-27b": "google/gemma-2-27b",
    "gemma2-9b": "google/gemma-2-9b-it",
    "gemma3-1b": "google/gemma-3-1b-it",
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma3-12b": "google/gemma-3-12b-it",
    "phi3-7b": "microsoft/Phi-3-small-8k-instruct",
    "phi3-3b": "microsoft/Phi-3-mini-4k-instruct",
    "phi3.5-3b": "microsoft/Phi-3.5-mini-instruct",
    "phi4-3b": "microsoft/Phi-4-mini-instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral-small": "mistralai/Mistral-Small-Instruct-2409",
    "mistral-large": "mistralai/Mistral-Large-Instruct-2407",
    "oss-20b": "openai/gpt-oss-20b",
}

# Datasets that expose a discrete extracted answer (used for argmin over extracted_answers).
EXTRACTED_ANSWER_DATASETS = {
    "gsm8k", "formal_logic", "arith_long", "pro_med", "mmlu_pro", "crux_eval"
}


# =====================================================================
# 2. ENVIRONMENT / SEEDING
# =====================================================================
def set_seed(seed_value=10):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


# =====================================================================
# 3. MODEL LOADING
# =====================================================================
def load_huggingface_model(model_name):
    hf_model = MODEL_PATH_DICT.get(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model,
        padding_side="left",
        trust_remote_code=True,
        use_fast=False if "falcon3" in hf_model else True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=config.hf_cache_dir,
    )
    return tokenizer, model


def load_model_from_path(model_name, device):
    if model_name not in MODEL_PATH_DICT:
        raise ValueError(f"Model {model_name} not supported")
    model_path = MODEL_PATH_DICT[model_name].lower()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
        use_fast=False if "falcon3" in model_path else True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        cache_dir=config.hf_cache_dir,
    )
    return model, tokenizer


# =====================================================================
# 4. TEXT CLEANING & ANSWER EXTRACTION
# =====================================================================
def clean_generation(text):
    for s in [
        "Q:", "A:", "question:", "answer:", "Question:", "Answer:",
        "Questions:", "questions:", "QUESTION:", "ANSWER:", ":",
    ]:
        if s in text:
            text = text.split(s)[0].rstrip()
    return text.strip()


def clean_answer(model_pred):
    """Extract last numeric answer from a generated trace (GSM8K-style)."""
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = len(preds) > 1
    pred = preds[1] if answer_flag else preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    pred = pred[0] if answer_flag else pred[-1]
    if pred[-1] == ".":
        pred = pred[:-1]
    return pred


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        return match.group(1).strip().replace(",", "")
    return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def extract_math_answer(full_ans_text: str):
    match = ANS_RE.search(full_ans_text)
    if match:
        return int(match.group(1).strip().replace(",", ""))
    return None


def extract_math_response(text, args):
    """Extract the last `{...}` chunk and parse it for math / MCQ datasets."""
    try:
        pred = re.findall(r"\{(.*?)\}", text)[-1]
        pred = pred.replace("final answer:", "").strip()

        if args.dataset in ["gsm8k", "arith_long"]:
            text = round(float(pred), 1)
        elif args.dataset in ["formal_logic", "pro_med", "mmlu_pro"]:
            if len(pred) == 0:
                text = ""
            elif len(pred) < 3:
                text = f"({pred[0]})"
            else:
                text = f"({pred[1]})"
    except Exception:
        text = ""
    return text


# ---------- Code-output extraction (CRUX-Eval) ----------
def extract_json_safe(text):
    """Extract the LAST valid JSON object that contains key 'answer'."""
    if not isinstance(text, str):
        return None

    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)

    decoder = json.JSONDecoder()
    idx, n, last_valid = 0, len(text), None

    while idx < n:
        if text[idx] == "{":
            try:
                obj, end = decoder.raw_decode(text[idx:])
                if isinstance(obj, dict) and "answer" in obj:
                    last_valid = obj
                idx += end
                continue
            except json.JSONDecodeError:
                pass
        idx += 1
    return last_valid


def try_parse_literal(x):
    if not isinstance(x, str):
        return x
    x = x.strip()
    try:
        return json.loads(x)
    except Exception:
        pass
    try:
        return ast.literal_eval(x)
    except Exception:
        return x


def simple_extract_answer(text):
    if not isinstance(text, str):
        return None

    patterns = [
        r"final answer\s*[:=]\s*([^\n]+)",
        r"the answer is\s*([^\n]+)",
        r"(?<!correct_)\banswer\s*[:=]\s*([^\n]+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            ans = re.split(r"\n|\.$", ans)[0]
            return try_parse_literal(ans)

    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return try_parse_literal(numbers[-1])
    return None


def normalize_answer(x):
    if x is None:
        return x
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()

    try:
        x = ast.literal_eval(x)
    except Exception:
        pass

    def convert_keys(obj):
        if isinstance(obj, dict):
            return {str(k): convert_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_keys(v) for v in obj]
        return obj

    if isinstance(x, (list, dict)):
        try:
            return json.dumps(convert_keys(x), sort_keys=True)
        except Exception:
            return json.dumps(str(x))

    if isinstance(x, str):
        return x.strip().strip("\"'").strip()
    return str(x)


def extract_code_response(prediction_str):
    pred = extract_json_safe(prediction_str)
    if pred and "answer" in pred:
        return normalize_answer(pred["answer"])
    return normalize_answer(simple_extract_answer(prediction_str))


def code_eval(a, b):
    if type(a) != type(b):
        return str(a) == str(b)
    return a == b


# =====================================================================
# 5. INSTRUCTION TEMPLATES
# =====================================================================
def get_instruction_suffix(args):
    if args.dataset in ["arithmetics", "arith_long"]:
        return (
            " Make sure to state your final answer in curly brackets at the very end "
            "of your response, just like: '{final answer: 12.34}'. Let's think step by step."
        )
    if args.dataset in ["gsm8k"]:
        return (
            " Make sure to state your final answer in curly brackets at the very end "
            "of your response, just like: '{final answer: 123}'. Let's think step by step."
        )
    if args.dataset in [
        "hellaswag", "pro_med", "formal_logic", "csqa", "hh_rlhf", "mmlu_pro"
    ]:
        return (
            " Make sure to state your final answer choice in curly brackets at the very "
            "end of your response, just like: '{final answer: (A)}'. Let's think step by step."
        )
    if args.dataset in ["cnn_daily"]:
        return ' Make sure to provide your summary after stating "# Summary # ".'
    if args.dataset in ["crux_eval"]:
        return (
            "You are given a Python function and some inputs. \n"
            "Your task is to determine the exact output of the function when called with those inputs.\n\n"
            "Think step by step inside your mind, but **DO NOT output any reasoning, explanation, or extra text**.\n"
            "Your response MUST be **ONLY** a valid JSON object in this exact format:\n"
            "{\"answer\": <final_output>}\n"
            "Do not include any markdown, code blocks, or additional text outside the JSON."
        )


# =====================================================================
# 6. DATASET LOADERS
# =====================================================================
def load_mmlu_pro(split="test", n_sample_per_cat=10):
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")[split]
    dataset = pd.DataFrame(dataset)

    selected_categories = dataset["category"].unique()
    selected_question_idx = []
    for cat in selected_categories:
        question_idx = dataset.loc[dataset["category"] == cat, "question_id"].tolist()
        selected_question_idx.extend(question_idx[:n_sample_per_cat])

    dataset = dataset[dataset["question_id"].isin(selected_question_idx)]

    questions, labels = [], []
    choices = "ABCDEFGHIJ"
    template = (
        "{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n(E) {}\n(F) {}\n(G) {}\n(H) {}\n(I) {}\n(J) {}\n\n"
    )
    for query, options, answer in zip(
        dataset["question"], dataset["options"], dataset["answer_index"]
    ):
        if len(options) != 10:
            continue
        question = template.format(query, *options)
        labels.append(f"({choices[int(answer)]})")
        questions.append(question)

    return questions, labels


# ---------- Black-box benchmark loaders (used by blackbox.py) ----------
def load_hle(split="test", max_samples=100):
    """Humanity's Last Exam — text-only, short-answer subset."""
    dataset = pd.DataFrame(load_dataset("cais/hle")[split])
    # Drop multimodal rows and keep short answers (<= 5 chars).
    df = dataset[dataset["image"] == ""]
    df = df[df["answer"].apply(lambda x: len(x) <= 5)]
    print(f"Loaded HLE dataset with {len(df)} samples after filtering.")

    questions = list(df["question"])
    labels = list(df["answer"])
    return questions[:max_samples], labels[:max_samples]


def load_gpqa(split="validation"):
    """GPQA-Diamond formatted as 4-way MCQ with the correct answer fixed at (A)."""
    dataset = pd.DataFrame(load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"])

    questions, labels = [], []
    template = "{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n\n"
    for ctx, A, B, C, D in zip(
        dataset["Question"], dataset["Correct Answer"],
        dataset["Incorrect Answer 1"], dataset["Incorrect Answer 2"],
        dataset["Incorrect Answer 3"],
    ):
        questions.append(template.format(ctx, A, B, C, D))
        labels.append("(A)")
    return questions, labels


def load_cruxeval(max_samples=100):
    ds = load_dataset("cruxeval-org/cruxeval")["test"]
    format_code = "Code: {code}\n\nInput: {input}"
    questions = [format_code.format(code=item["code"], input=item["input"]) for item in ds]
    answers = [item["output"] for item in ds]
    return questions[:max_samples], answers[:max_samples]


def load_bigbenchhard_nav(split="validation"):
    dataset = pd.DataFrame(load_dataset("maveriq/bigbenchhard", "navigate")["train"])
    return list(dataset["input"]), list(dataset["target"])


def load_bigbenchhard_date(split="validation"):
    dataset = pd.DataFrame(load_dataset("maveriq/bigbenchhard", "date_understanding")["train"])
    return list(dataset["input"]), list(dataset["target"])


def extract_answer_response(text, data_type="math"):
    """Parse a generated string by black-box data-type. Mirrors `extract_math_response`
    but dispatches on a free-form `data_type` tag instead of `args.dataset`.
    """
    try:
        pred = re.findall(r"\{(.*?)\}", text)[-1]
        pred = pred.replace("final answer:", "").strip()

        if data_type == "math":
            text = int(pred)
        elif data_type in ["mc", "bbh_date"]:
            if len(pred) == 0:
                text = ""
            elif len(pred) < 3:
                text = f"({pred[0]})"
            else:
                text = f"({pred[1]})"
        elif data_type in ["hle", "bbh_nav"]:
            text = pred
        else:
            raise ValueError("Unknown data type")
    except Exception:
        text = ""
    return text


def load_data(args, split=None, easy=False):
    data_size = args.data_size
    num_params = 4 if easy else 6

    rng_seed = 0 if split == "train" else 1
    x = np.random.default_rng(rng_seed).integers(0, 30, size=num_params * data_size)

    X, Y = [], []
    for i in range(0, num_params * data_size, num_params):
        if easy:
            a, b, c, d = x[i:i + 4]
            X.append(f"What is the result of {a}+{b}*{c}-{d}?")
            Y.append(a + b * c - d)
        else:
            a, b, c, d, e, f = x[i:i + 6]
            if f == 0:
                f = 1
            X.append(f"What is the result of {a}+{b}*{c}+{d}-{e}÷{f}?")
            Y.append(a + b * c + d - e / f)
    return X, Y


def parse_dataset(args):
    stories = None

    if args.dataset in ["sciq", "nq"]:
        question_file = f"{config.data_dir}/{args.dataset}.txt"
        if not os.path.exists(question_file):
            raise FileNotFoundError(f"Question file not found: {question_file}")

        questions, answers = [], []
        with open(question_file, "r", encoding="utf-8") as f:
            blocks = f.read().strip().split("\n\n")
        for item in blocks:
            lines = item.strip().split("\n")
            if len(lines) < 2:
                continue
            questions.append(lines[0].strip())
            answers.append([a.strip() for a in lines[1].split(";") if a.strip()][0])

    elif args.dataset == "svamp":
        ds = load_dataset("ChilleD/SVAMP")["test"]
        questions = [item["question_concat"] for item in ds]
        answers = [item["Answer"] for item in ds]

    elif args.dataset == "arith":
        ds = load_dataset(
            "json",
            data_files="https://huggingface.co/datasets/EleutherAI/arithmetic/resolve/main/data/single_digit_three_ops.jsonl",
        )["train"]
        questions = [item["context"] for item in ds]
        answers = [item["completion"] for item in ds]

    elif args.dataset == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
        questions = [item["Pre-Revision Question"] for item in ds]
        answers = [item["Pre-Revision Correct Answer"] for item in ds]

    elif args.dataset == "trivia_qa":
        val_data = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        train_data = load_dataset("trivia_qa", "rc.nocontext", split="train")
        data_for_few_shot_prompt = train_data.select(range(0, 10))

        few_shot_prompt = "This is a bot that correctly answers questions. \n"
        for sample in data_for_few_shot_prompt:
            few_shot_prompt += (
                "Question: " + sample["question"]
                + " Answer: " + sample["answer"]["value"] + " "
            )
        questions = [item["question"] for item in val_data]
        answers = [item["answer"]["value"] for item in val_data]

    elif args.dataset == "truthful_qa":
        val_data = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
        data_for_few_shot_prompt = val_data.select(range(0, 10))

        few_shot_prompt = "This is a bot that correctly answers questions. \n"
        for sample in data_for_few_shot_prompt:
            few_shot_prompt += (
                "Question: " + sample["question"]
                + " Answer: " + sample["best_answer"] + " "
            )
        questions = [item["question"] for item in val_data.select(range(10, len(val_data)))]
        answers = [item["best_answer"] for item in val_data.select(range(10, len(val_data)))]

    elif args.dataset == "coqa":
        with open(f"{config.data_dir}/coqa.json", "r") as infile:
            data = json.load(infile)["data"]

        questions, answers, stories = [], [], []
        for sample in data:
            story = sample["story"]
            list_questions = sample["questions"]
            list_answers = sample["answers"]
            for question_index, question in enumerate(list_questions):
                question = question["input_text"]
                answer = list_answers[question_index]["input_text"]
                story = story + "\nQuestion: " + question + "\nAnswer: " + answer
                questions.append(story + f"\nQuestion: {question}\nAnswer: ")
                answers.append(answer)
            stories.append(story)

    elif args.dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main")["test"]
        questions = [item["question"] for item in ds]
        answers = [item["answer"] for item in ds]

    elif args.dataset == "arith_long":
        num_params, data_size = 6, 1000
        x = np.random.default_rng(1).integers(0, 30, size=num_params * data_size)
        questions, answers = [], []
        for i in range(0, num_params * data_size, num_params):
            a, b, c, d, e, f = x[i:i + 6]
            if f == 0:
                f = 1
            questions.append(f"What is the result of {a}+{b}*{c}+{d}-{e}÷{f}?")
            answers.append(str(a + b * c + d - e / f))

    elif args.dataset in ["formal_logic", "pro_med"]:
        mmlu_subject = "formal_logic" if args.dataset == "formal_logic" else "professional_medicine"
        ds = pd.DataFrame(load_dataset("cais/mmlu", mmlu_subject)["test"])

        questions, answers = [], []
        choices = "ABCD"
        for query, options, answer in zip(ds["question"], ds["choices"], ds["answer"]):
            if len(options) != 4:
                continue
            questions.append(
                "{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n\n".format(query, *options)
            )
            answers.append(f"({choices[int(answer)]})")

    elif args.dataset == "crux_eval":
        ds = load_dataset("cruxeval-org/cruxeval")["test"]
        questions = [
            f"Code: {item['code']}\n\nInput: {item['input']}" for item in ds
        ]
        answers = [item["output"] for item in ds]

    elif args.dataset == "mmlu_pro":
        questions, answers = load_mmlu_pro(split="test", n_sample_per_cat=10)

    else:
        raise ValueError(f"Dataset {args.dataset} not supported for parsing.")

    # Build few-shot prompt
    n_few = min(args.few_shot_num, len(questions))
    if args.dataset in [
        "gsm8k", "formal_logic", "arith_long", "pro_med", "crux_eval", "mmlu_pro"
    ]:
        few_shot_prompt = get_instruction_suffix(args=args)
    elif args.dataset == "coqa":
        few_shot_prompt = (
            f"This is a bot that correctly answers questions based on the provided context."
            f"\n\n{stories[0]}\n\n"
        )
    elif args.dataset in ["trivia_qa", "truthful_qa"]:
        pass  # already constructed above
    else:
        few_shot_prompt = "This is a bot that correctly answers questions.\n"
        for i in range(n_few):
            if args.dataset in ["sciq", "nq"]:
                few_shot_prompt += f"Question: {questions[i]}\nAnswer: {answers[i]}\n\n"
            elif args.dataset == "arith":
                few_shot_prompt += f"{questions[i]}\n{answers[i]}\n\n"
            else:
                few_shot_prompt += f"Question: {questions[i]}\nAnswer: {answers[i]}\n\n"

    processed_dataset = []
    if args.dataset == "coqa":
        for i in range(n_few, len(questions)):
            processed_dataset.append({
                "question": questions[i],
                "answer": answers[i],
                "prompt": few_shot_prompt + questions[i],
            })
    elif args.dataset == "arith":
        for i in range(n_few, len(questions)):
            processed_dataset.append({
                "question": questions[i],
                "answer": answers[i],
                "prompt": few_shot_prompt + f"{questions[i]}\n",
            })
    elif args.dataset in [
        "gsm8k", "formal_logic", "arith_long", "pro_med", "crux_eval", "mmlu_pro"
    ]:
        for i in range(len(questions)):
            processed_dataset.append({
                "question": questions[i],
                "answer": answers[i],
                "prompt": f"Question: {questions[i]}\n" + few_shot_prompt,
            })
    else:
        for i in range(n_few, len(questions)):
            processed_dataset.append({
                "question": questions[i],
                "answer": answers[i],
                "prompt": few_shot_prompt + f"Question: {questions[i]}\nAnswer:",
            })

    return processed_dataset


# =====================================================================
# 7. GENERATION + NLL
# =====================================================================
def flatten_logprobs(logprobs):
    """Flatten nested list[dict[token_id -> Logprob]] into a list of floats."""
    flat = []
    if not logprobs:
        return flat
    if isinstance(logprobs, list):
        for step in logprobs:
            if isinstance(step, dict):
                flat.extend([v.logprob for v in step.values()])
    elif isinstance(logprobs, dict):
        flat.extend([v.logprob for v in logprobs.values()])
    return flat


def generate_sequences(llm, dataset, rouge, args):
    print("--- GENERATION PARAMETERS ---")
    print("Dataset:", args.dataset)
    print("Model:", args.model)

    greedy_params = SamplingParams(
        max_tokens=args.max_new_tokens, temperature=0, n=1, logprobs=1
    )
    multinomial_params = SamplingParams(
        max_tokens=args.max_new_tokens, temperature=1, n=args.n_samples, logprobs=1
    )

    sequences = []
    for i, batch in enumerate(tqdm(dataset)):
        prompt = batch["prompt"]
        question = batch["question"]
        answer = batch["answer"].strip()

        # === GREEDY ===
        greedy_out = llm.generate(prompt, sampling_params=greedy_params, use_tqdm=False)[0].outputs[0]
        greedy_text_raw = greedy_out.text.strip()

        if args.dataset in ["gsm8k", "formal_logic", "arith_long", "pro_med", "mmlu_pro"]:
            greedy_text = extract_math_response(text=greedy_text_raw, args=args)
            if args.dataset in ["gsm8k"]:
                answer = extract_math_answer(answer)
            elif args.dataset in ["arith_long"]:
                answer = float(answer)
        elif args.dataset == "crux_eval":
            greedy_text = extract_code_response(greedy_text_raw)
            answer = normalize_answer(answer)
        else:
            greedy_text = clean_generation(greedy_text_raw)

        greedy_logprobs = greedy_out.logprobs

        llm_label = None
        if args.dataset in ["svamp", "arith"]:
            eval_score = compute_label(greedy_text, answer, eval_method="exact_match")
        elif args.dataset in ["gsm8k", "arith_long"]:
            eval_score = int(greedy_text == np.round(answer, 1))
        elif args.dataset in ["formal_logic", "pro_med", "mmlu_pro"]:
            eval_score = int(greedy_text == answer)
        elif args.dataset == "crux_eval":
            eval_score = code_eval(greedy_text, answer)
        else:
            eval_score = compute_label(greedy_text, answer, rouge=rouge, eval_method="rougeL")
            llm_label = compute_label(
                greedy_text, answer, question=question,
                eval_method="llm_eval", api_type=args.api_type,
            )

        # === MULTINOMIAL ===
        sampled_outputs = llm.generate(prompt, sampling_params=multinomial_params, use_tqdm=False)[0].outputs
        generated_texts = [o.text for o in sampled_outputs]
        generation_logprobs = [o.logprobs for o in sampled_outputs]

        cleaned = [clean_generation(g) for g in generated_texts]
        if args.dataset in ["gsm8k", "formal_logic", "arith_long", "pro_med", "mmlu_pro"]:
            extracted_answers = [extract_math_response(text=g, args=args) for g in generated_texts]
        elif args.dataset == "crux_eval":
            extracted_answers = [extract_code_response(g) for g in generated_texts]
        else:
            extracted_answers = [clean_generation(g) for g in generated_texts]

        samples_avg_nll, samples_nll = [], []
        for sample in generation_logprobs:
            flat = flatten_logprobs(sample)
            samples_avg_nll.append(-np.mean(flat) if len(flat) > 0 else np.nan)
            samples_nll.append(-np.sum(flat) if len(flat) > 0 else np.nan)

        greedy_flat = flatten_logprobs(greedy_logprobs)
        greedy_avg_nll = -np.mean(greedy_flat) if greedy_flat else np.nan
        greedy_nll = -np.sum(greedy_flat) if greedy_flat else np.nan

        if i < 5:
            print("Prompt:", prompt)
            print("Question:", question)
            print("Answer:", answer)
            print("Greedy text:", greedy_text_raw)
            print("Cleaned greedy text:", greedy_text)
            print("Generated texts:", generated_texts)
            print("Extracted answers:", extracted_answers)
            print("Samples avg NLL:", samples_avg_nll)
            print("Eval score:", eval_score)
            print("---")

        sequences.append({
            "id": f"{args.dataset}_{i}",
            "prompt": prompt,
            "question": question,
            "answer": answer,
            "generated_texts": generated_texts,
            "cleaned_generated_texts": cleaned,
            "extracted_answers": extracted_answers,
            "samples_nll": samples_nll,
            "samples_avg_nll": samples_avg_nll,
            "greedy_text": greedy_text,
            "greedy_text_raw": greedy_text_raw,
            "greedy_nll": greedy_nll,
            "greedy_avg_nll": greedy_avg_nll,
            "eval_score": eval_score,
            "llm_label": llm_label,
            "max_new_tokens": args.max_new_tokens,
        })

    return sequences


# =====================================================================
# 8. LLM-BASED EVALUATION
# =====================================================================
def cohere_evaluate(message):
    cohere_client = cohere.ClientV2(api_key=api_key.cohere_api_key)
    try:
        response = cohere_client.chat(messages=message, model=api_key.cohere_model)
        return response.message.content[0].text
    except Exception as e:
        print("Evaluation Error: ", e)
        print(message)
        return None


def gemini_evaluate(message):
    gemini_client = genai.Client(api_key=api_key.gemini_api_key)
    try:
        message = message[0]["content"] if message else ""
        response = gemini_client.models.generate_content(
            model=api_key.gemini_model, contents=message
        )
        return response.text
    except Exception as e:
        print("Evaluation Error: ", e)
        print(message)
        return None


def compute_label(generation, ground_truth, question=None, rouge=None,
                  eval_method="exact_match", api_type="cohere"):
    if eval_method == "exact_match":
        return generation == ground_truth

    if eval_method == "rougeL":
        rouge_output = rouge.compute(predictions=[generation], references=[ground_truth])
        return rouge_output["rougeL"]

    if eval_method == "llm_eval":
        message = [{
            "role": "user",
            "content": (
                f"You are a helpful and precise assistant for checking the quality of the answer.\n"
                f"Question: {question}\n"
                f"Ground Truth: {ground_truth}\n"
                f"Answer To Be Evaluated: {generation}\n"
                "Is the answer correct? Please answer with 'Yes' or 'No'."
            ),
        }]
        try:
            if api_type == "cohere":
                llm_response = cohere_evaluate(message)
            elif api_type == "gemini":
                llm_response = gemini_evaluate(message)
            else:
                raise ValueError(f"Unsupported API type: {api_type}")
        except Exception as e:
            print(f"Error during LLM evaluation: {e}")
            llm_response = None
        return 1 if llm_response and llm_response.strip().lower().startswith("yes") else 0

    raise ValueError(f"Unsupported eval method: {eval_method}")


def evaluation_sample(dataset, text, answer, rouge, question=None,
                      eval_method="rougeL", api_type="cohere", threshold=0.3):
    """Score one (prediction, answer) pair into a binary accuracy."""
    if dataset in ["svamp", "arith"]:
        eval_score = compute_label(text, answer, eval_method="exact_match")
    elif dataset in ["gsm8k", "arith_long"]:
        eval_score = int(text == np.round(answer, 1))
    elif dataset in ["formal_logic", "pro_med", "mmlu_pro"]:
        eval_score = int(text == answer)
    elif dataset == "crux_eval":
        eval_score = code_eval(text, answer)
    else:
        eval_score = compute_label(
            text, answer, question=question, eval_method=eval_method,
            rouge=rouge, api_type=api_type,
        )

    if dataset in [
        "gsm8k", "svamp", "arith", "arith_long",
        "formal_logic", "pro_med", "mmlu_pro", "crux_eval"
    ]:
        return int(eval_score == 1.0)
    if eval_method == "rougeL":
        return int(eval_score > threshold)
    return int(eval_score)


# =====================================================================
# 9. CALIBRATION HELPERS
# =====================================================================
def compute_ece(confidence, labels, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            acc = np.mean(1 - labels[mask])
            conf = np.mean(confidence[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(labels)
    return ece


def minmax_normalize(x):
    x = np.array(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


# =====================================================================
# 10. DISTANCE / RDS METRICS
# =====================================================================
def compute_weighted_mean(embeddings, weights):
    """Weighted Fréchet mean (in Euclidean space) of a set of embeddings."""
    weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(embeddings.device)
    return (weights * embeddings).sum(dim=0)


def compute_metrics(embeddings, mean, weights, p=1):
    """Weighted Lp distance from each embedding to a central point (returns scalar)."""
    device = embeddings.device
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    w = w / w.sum()
    diffs = embeddings - mean
    norms = torch.norm(diffs, p=p, dim=1)
    return (w * norms).sum().item()


def compute_rds(embeddings, center):
    """L2 distance from each embedding to `center`."""
    diffs = embeddings - center.unsqueeze(0)
    return torch.norm(diffs, p=2, dim=-1)


def compute_rds_raw(answers, weights=None):
    """RDS on raw numeric answers (for math datasets)."""
    answers = np.array(answers)
    mean_answer = np.average(answers, weights=weights)
    return np.abs(answers - mean_answer)


def compute_rds_raw_medoid_weighted(answers, weights=None):
    """Distance to the (weighted) medoid in raw numeric answer space."""
    answers = np.array(answers)
    N = len(answers)
    weights = np.ones(N) if weights is None else np.array(weights)

    dists = np.abs(answers[:, None] - answers[None, :])
    total_dist = (dists * weights[None, :]).sum(axis=1)
    medoid = answers[np.argmin(total_dist)]
    return np.abs(answers - medoid)


def compute_medoid(points, weights=None):
    """Weighted medoid in embedding space."""
    if points.dim() == 1:
        return points
    if points.size(0) == 1:
        return points[0]

    dists = torch.cdist(points, points, p=2)
    if weights is not None:
        total_dist = (dists * weights.view(1, -1)).sum(dim=1)
    else:
        total_dist = dists.sum(dim=1)
    return points[torch.argmin(total_dist)]


def geometric_median(points, eps=1e-5, max_iter=100):
    """Weiszfeld iteration for the geometric median of a point cloud."""
    median = points.mean(dim=0)
    for _ in range(max_iter):
        diffs = points - median
        distances = torch.norm(diffs, p=2, dim=1).clamp(min=eps)
        weights = 1.0 / distances
        new_median = (points * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
        if torch.norm(new_median - median) < eps:
            break
        median = new_median
    return median


# =====================================================================
# 11. SELF-CERTAINTY
# =====================================================================
def confidence_logprob_sum(logprob_sum: torch.Tensor, attention_mask: torch.Tensor, V: int):
    logprob_sum = logprob_sum.contiguous()
    attention_mask = attention_mask.contiguous()
    V_tensor = torch.tensor(V, dtype=logprob_sum.dtype, device=logprob_sum.device)
    conf = -1 / V * logprob_sum - torch.log(V_tensor)
    valid_conf = conf * attention_mask
    return (valid_conf.sum(dim=-1) / attention_mask.sum(dim=-1)).tolist()


def get_self_certainty_sample(all_confidences, answers, power=0.3):
    sorted_indices = sorted(range(len(all_confidences)), key=lambda k: all_confidences[k], reverse=True)
    votes_per_output = [(len(all_confidences) - rank) ** power for rank in range(len(all_confidences))]

    votes_map = {sorted_indices[i]: votes_per_output[i] for i in range(len(sorted_indices))}
    votes = [0 for _ in range(len(all_confidences))]
    for i in range(len(all_confidences)):
        answer_i = answers[i]
        if answer_i is None:
            continue
        find_answer = False
        for j in range(i):
            answer_j = answers[j]
            if answer_j is None:
                continue
            if answer_i == answer_j:
                votes[j] += votes_map[i]
                find_answer = True
                break
        if not find_answer:
            votes[i] += votes_map[i]

    best_index = votes.index(max(votes))
    return answers[best_index]


@torch.no_grad()
def compute_self_certainty_scores(
    model_dir: str,
    prompts: list,
    generated_texts_list: list,
    batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_length: int = 2048,
):
    """Reference: https://github.com/backprop07/Self-Certainty/blob/main/src/confidence_list.py"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    all_confidences = []

    for prompt, generated_texts in tqdm(
        zip(prompts, generated_texts_list), total=len(prompts)
    ):
        prompt_enc = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=max_length, add_special_tokens=False,
        ).to(device)
        input_ids = prompt_enc.input_ids[0]
        input_mask = prompt_enc.attention_mask[0]
        input_len = input_mask.sum().item()

        confidences = [None] * len(generated_texts)

        # Length-based bucketing to avoid OOM
        groups = {"small": [], "medium": [], "large": []}
        indices = list(range(len(generated_texts)))
        for i, text in enumerate(generated_texts):
            l = len(text)
            if l > 6144:
                groups["large"].append(text)
            elif l > 3072:
                groups["medium"].append(text)
            else:
                groups["small"].append(text)

        group_bs = {
            "small": batch_size,
            "medium": max(1, batch_size // 2),
            "large": max(1, batch_size // 4),
        }

        for group_name in ["small", "medium", "large"]:
            texts = groups[group_name]
            if not texts:
                continue

            group_indices = [
                indices[i] for i in range(len(indices))
                if generated_texts[indices[i]] in texts
            ]
            bs = group_bs[group_name]

            out_enc = tokenizer(
                texts, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            out_ids = out_enc.input_ids
            out_mask = out_enc.attention_mask

            full_ids = torch.cat(
                [input_ids.unsqueeze(0).repeat(len(texts), 1), out_ids], dim=1
            ).long()
            full_mask = torch.cat(
                [input_mask.unsqueeze(0).repeat(len(texts), 1), out_mask], dim=1
            ).long()

            group_confs = []
            for i in range(0, len(texts), bs):
                j = i + bs
                logits = model(full_ids[i:j], attention_mask=full_mask[i:j]).logits
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    batch_logprob_sum = logits[:, input_len:, :]
                    batch_logprob_sum = F.log_softmax(batch_logprob_sum, dim=-1)
                    batch_logprob_sum = batch_logprob_sum.sum(dim=-1).to(device).to(torch.float32)

                vocab_size = getattr(model.config, "vocab_size", None)
                if vocab_size is None:
                    vocab_size = model.get_input_embeddings().weight.shape[0]
                group_confs.extend(confidence_logprob_sum(
                    batch_logprob_sum, out_mask[i:j], vocab_size
                ))

            for conf, orig_idx in zip(group_confs, group_indices):
                confidences[orig_idx] = float(conf)

        all_confidences.append(confidences)

    return all_confidences


# =====================================================================
# 12. MODEX
# =====================================================================
def compute_semantics_adjacency_matrix(agent_names, agent_responses, emb_encoder):
    texts = [str(agent_responses[name]) for name in agent_names]
    embeddings = emb_encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return np.dot(embeddings, embeddings.T)


def compute_text_adjacency_matrix(agent_names, agent_responses):
    """Jaccard similarity on cached uni/bi/tri-grams."""
    n = len(agent_names)
    A = np.zeros((n, n))

    def get_ngrams(text, ngram=3):
        tokens = str(text).lower().split()
        ngrams = set()
        for k in range(1, ngram + 1):
            for i in range(len(tokens) - k + 1):
                ngrams.add(tuple(tokens[i:i + k]))
        return ngrams

    cached = {name: get_ngrams(agent_responses[name]) for name in agent_names}

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
    """Simple spectral 2-way cut using the Fiedler vector of the Laplacian."""
    D = np.diag(np.sum(A, axis=1))
    L = D - A

    _, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]

    group1 = [names[i] for i in range(len(names)) if fiedler_vector[i] >= 0]
    group2 = [names[i] for i in range(len(names)) if fiedler_vector[i] < 0]
    return {"groups": {"group_1": group1, "group_2": group2}}


def goodness_of_cut_func(A, group_indices, n_group_indices, method="conductance"):
    cut = float(np.sum(A[np.ix_(group_indices, n_group_indices)]))
    if method == "conductance":
        vol_s = float(np.sum(A[group_indices, :]) - len(group_indices))
        vol_sbar = float(np.sum(A[n_group_indices, :]) - len(n_group_indices))
        return cut / (min(vol_s, vol_sbar) + 1e-10)
    if method == "cutratio":
        return cut / (np.sum(A) - A.shape[0])
    vol_s = float(np.sum(A[group_indices, :]) - len(group_indices))
    vol_sbar = float(np.sum(A[n_group_indices, :]) - len(n_group_indices))
    return (cut / vol_s) + (cut / vol_sbar)


def modex_select(texts, adjacency="text", tau=0.8,
                 goodness_of_cut="conductance", emb_encoder=None):
    """Evaluator-free Best-of-N via recursive spectral graph cuts."""
    if len(texts) <= 1:
        return 0

    n = len(texts)
    agent_names = [f"agent_{i}" for i in range(n)]
    agent_responses = dict(zip(agent_names, texts))

    if adjacency == "semantics":
        A = compute_semantics_adjacency_matrix(agent_names, agent_responses, emb_encoder)
    elif adjacency == "text":
        A = compute_text_adjacency_matrix(agent_names, agent_responses)
    elif adjacency == "both":
        A_sem = compute_semantics_adjacency_matrix(agent_names, agent_responses, emb_encoder)
        A_text = compute_text_adjacency_matrix(agent_names, agent_responses)
        A = 0.5 * (A_sem + A_text)
    else:
        raise ValueError("adjacency must be 'semantics', 'text', or 'both'")

    _A = A.copy()
    current_names = agent_names.copy()

    while True:
        info = graph_cut(_A, current_names)
        g1 = info["groups"]["group_1"]
        g2 = info["groups"]["group_2"]
        group = g1 if len(g1) >= len(g2) else g2
        n_group = g2 if len(g1) >= len(g2) else g1

        prev_n = len(current_names)
        group_indices = [current_names.index(name) for name in group]
        n_group_indices = [current_names.index(name) for name in n_group]

        phi = goodness_of_cut_func(_A, group_indices, n_group_indices, goodness_of_cut)
        if phi >= tau:
            degrees = np.sum(_A, axis=1)
            best_name = current_names[int(np.argmax(degrees))]
            return agent_names.index(best_name)

        _A = _A[np.ix_(group_indices, group_indices)]
        current_names = [current_names[i] for i in group_indices]

        if len(current_names) == prev_n or len(current_names) <= 1:
            break

    return agent_names.index(np.random.choice(current_names))


# =====================================================================
# 13. BLACK-BOX PROMPT SUFFIXES
# =====================================================================
SUFFIX_MATH = (
    " Make sure to state your final answer in curly brackets at the very end of "
    "your response, just like: '{final answer: 12.34}'. Let's think step by step."
)
SUFFIX_MC = (
    " Make sure to state your final answer choice in curly brackets at the very "
    "end of your response, just like: '{final answer: (A)}'. Let's think step by step."
)
SUFFIX_HLE = (
    " Make sure to state your final answer in curly brackets at the very end of "
    "your response, just like: '{final answer: Z+Z+Z+Z+Z}'. Let's think step by step."
)
SUFFIX_BBH_DATE = (
    "Do not include any reasoning steps. Make sure to state your final answer in "
    "curly brackets at the very end of your response, just like: '{final answer: (A)}'."
)
SUFFIX_BBH_NAV = (
    "Do not include any reasoning steps. Make sure to state your final answer in "
    "curly brackets at the very end of your response, just like: '{final answer: Yes}'."
)
SUFFIX_CODE = (
    "You are given a Python function and some inputs. \n"
    "Your task is to determine the exact output of the function when called with those inputs.\n\n"
    "Think step by step inside your mind, but **DO NOT output any reasoning, explanation, or extra text**.\n"
    "Your response MUST be **ONLY** a valid JSON object in this exact format:\n"
    "{\"answer\": <final_output>}\n"
    "Do not include any markdown, code blocks, or additional text outside the JSON."
)


def get_blackbox_dataset(dataset_name):
    """Return `(questions, labels, suffix, data_type)` for a black-box benchmark."""
    if dataset_name == "aime25":
        ds = load_dataset("MathArena/aime_2025")["train"]
        questions = [item["problem"] for item in ds]
        labels = [item["answer"] for item in ds]
        return questions, labels, SUFFIX_MATH, "math"

    if dataset_name == "mmlu_pro":
        questions, labels = load_mmlu_pro(split="test")
        return questions, labels, SUFFIX_MC, "mc"

    if dataset_name == "hle":
        questions, labels = load_hle(split="test")
        return questions, labels, SUFFIX_HLE, "hle"

    if dataset_name == "gpqa":
        questions, labels = load_gpqa(split="train")
        return questions, labels, SUFFIX_MC, "mc"

    if dataset_name == "bbh_nav":
        questions, labels = load_bigbenchhard_nav(split="train")
        return questions, labels, SUFFIX_BBH_NAV, "bbh_nav"

    if dataset_name == "bbh_date":
        questions, labels = load_bigbenchhard_date(split="train")
        return questions, labels, SUFFIX_BBH_DATE, "bbh_date"

    if dataset_name == "cruxeval":
        questions, labels = load_cruxeval()
        return questions, labels, SUFFIX_CODE, "code"

    raise ValueError(f"Unsupported black-box dataset: {dataset_name}")
