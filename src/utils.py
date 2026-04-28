import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset
import re, json, random, os
import cohere
from google import genai
from vllm import SamplingParams

import logging
logging.basicConfig(level=logging.ERROR)

from . import config, api_key

ANSWER_TRIGGER = 'The answer is'
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

### --------------------------------
### Model & preprocessing utils
### --------------------------------
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
        cache_dir=config.hf_cache_dir
    )
    
    return tokenizer, model

def set_seed(seed_value=10):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def flatten_logprobs(logprobs):
    """
    Flattens nested list[dict[token_id -> Logprob]] into a list of floats.
    """
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


# -----------------------------------------------
# CODE EXTRACTION UTILS
# -----------------------------------------------
import re
import json
import ast


# =========================
# 1. JSON extraction (robust)
# =========================
def extract_json_safe(text):
    """
    Extract the LAST valid JSON object that contains key 'answer'.
    Uses JSONDecoder to avoid brace-matching bugs.
    """
    if not isinstance(text, str):
        return None

    # remove markdown fences
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```', '', text)

    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)

    last_valid = None

    while idx < n:
        if text[idx] == "{":
            try:
                obj, end = decoder.raw_decode(text[idx:])
                if isinstance(obj, dict) and "answer" in obj:
                    last_valid = obj  # keep last match
                idx += end
                continue
            except json.JSONDecodeError:
                pass
        idx += 1

    return last_valid


# =========================
# 2. Literal parsing
# =========================
def try_parse_literal(x):
    """
    Convert string → Python object (list, int, float, etc.)
    """
    if not isinstance(x, str):
        return x

    x = x.strip()

    # try JSON first
    try:
        return json.loads(x)
    except:
        pass

    # fallback to python literal
    try:
        return ast.literal_eval(x)
    except:
        return x


# =========================
# 3. Regex fallback
# =========================
def simple_extract_answer(text):
    """
    Extract answer using regex heuristics.
    Non-greedy + safer patterns.
    """
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

    # fallback: last number
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return try_parse_literal(numbers[-1])

    return None


# =========================
# 4. Normalize
# =========================
def normalize_answer(x):
    if x is None:
        return x

    # convert tensors / weird types
    if not isinstance(x, str):
        x = str(x)

    x = x.strip()

    # 1. try parse as python literal ('abc', ["a"], {'a':1}, etc.)
    try:
        x = ast.literal_eval(x)
    except:
        pass

    # helper: convert all dict keys to string (recursive)
    def convert_keys(obj):
        if isinstance(obj, dict):
            return {str(k): convert_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys(v) for v in obj]
        else:
            return obj

    # 2. if list/dict → stringify stable
    if isinstance(x, (list, dict)):
        try:
            x = convert_keys(x)
            return json.dumps(x, sort_keys=True)
        except Exception:
            # fallback: avoid crash
            return json.dumps(str(x))

    # 3. final cleanup for strings
    if isinstance(x, str):
        return x.strip().strip('"\'').strip()

    return str(x)


# =========================
# 5. Main API
# =========================
def extract_code_response(prediction_str):
    """
    Main extraction pipeline:
    1. JSON (preferred)
    2. Regex fallback
    """

    # 1. Try JSON
    pred = extract_json_safe(prediction_str)
    if pred and "answer" in pred:
        return normalize_answer(pred["answer"])

    # 2. Fallback regex
    ans = simple_extract_answer(prediction_str)
    return normalize_answer(ans)


def code_eval(a, b):
    if type(a) != type(b):
        return str(a) == str(b)
    return a == b


def extract_math_response(text, args):
    try:
        pred = re.findall(r"\{(.*?)\}", text)[-1]
        pred = pred.replace("final answer:", "").strip()
        
        if args.dataset in ['gsm8k', 'arith_long']:
            pred = float(pred)
            text = round(pred, 1)
        
        elif args.dataset in ['formal_logic', 'pro_med', 'mmlu_pro']:
            if len(pred) == 0:
                text = ""
            elif len(pred) < 3:
                pred = pred[0]
                text = f"({pred})"
            else:
                pred = pred[1]
                text = f"({pred})"

    except :
        text =  ""
        
    return text

def clean_generation(text):
    for s in ['Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:', 'ANSWER:', ':']:
        if s in text:
            text = text.split(s)[0].rstrip()
    return text.strip()


def load_model_from_path(model_name, device):
    if model_name not in MODEL_PATH_DICT:
        raise ValueError(f"Model {model_name} not supported")
    model_path = MODEL_PATH_DICT[model_name].lower()

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
        use_fast=False if "falcon3" in model_path else True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        cache_dir=config.hf_cache_dir
    )

    return model, tokenizer 


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

# --------------------------------
# DATASET PARSING
# --------------------------------
def parse_dataset(args):
    
    if args.dataset in ['sciq', 'nq']:
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
            question = lines[0].strip()
            ans = [a.strip() for a in lines[1].split(";") if a.strip()][0]
            questions.append(question)
            answers.append(ans)

    elif args.dataset == 'svamp':

        ds = load_dataset("ChilleD/SVAMP")['test']
        questions = [item['question_concat'] for item in ds]
        answers = [item['Answer'] for item in ds]
    
    elif args.dataset == 'arith': # 0.5 fraction of data
        
        ds = load_dataset(
            "json", 
            data_files="https://huggingface.co/datasets/EleutherAI/arithmetic/resolve/main/data/single_digit_three_ops.jsonl"
            )['train']
        questions = [item['context'] for item in ds]
        answers = [item['completion'] for item in ds]
        
    elif args.dataset == 'gpqa': # 1 fraction of data
        
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")['train']
        questions = [item['Pre-Revision Question'] for item in ds]
        answers = [item['Pre-Revision Correct Answer'] for item in ds]
    
    elif args.dataset == 'trivia_qa':
        val_data = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        train_data = load_dataset("trivia_qa", "rc.nocontext", split="train")
        data_for_few_shot_prompt = train_data.select(range(0, 10))

        few_shot_prompt = 'This is a bot that correctly answers questions. \n'
        for sample in data_for_few_shot_prompt:
            few_shot_prompt += 'Question: ' + sample['question'] + ' Answer: ' + sample['answer']['value'] + ' '
            
        questions = [item['question'] for item in val_data]
        answers = [item['answer']['value'] for item in val_data] 
    
    elif args.dataset == 'truthful_qa':
        val_data = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
        data_for_few_shot_prompt = val_data.select(range(0, 10))

        few_shot_prompt = 'This is a bot that correctly answers questions. \n'
        for sample in data_for_few_shot_prompt:
            few_shot_prompt += 'Question: ' + sample['question'] + ' Answer: ' + sample['best_answer'] + ' '
            
        questions = [item['question'] for item in val_data.select(range(10, len(val_data)))]
        answers = [item['best_answer'] for item in val_data.select(range(10, len(val_data)))]

    elif args.dataset == 'coqa':
        with open(f'{config.data_dir}/coqa.json', 'r') as infile:
            data = json.load(infile)['data']

        questions, answers, stories = [], [], []

        for sample in data:
            story = sample['story']
            list_questions = sample['questions']
            list_answers = sample['answers']
            for question_index, question in enumerate(list_questions):
                
                question = question['input_text']
                answer = list_answers[question_index]['input_text']
                story = story + '\nQuestion: ' + question + '\nAnswer: ' + answer

                questions.append(story + f'\nQuestion: {question}\nAnswer: ')
                answers.append(answer)
                
            stories.append(story)

    elif args.dataset == 'gsm8k':
        ds = load_dataset("gsm8k", "main")['test']
        questions = [item['question'] for item in ds]
        answers = [item['answer'] for item in ds]
    
    elif args.dataset == 'arith_long':
        num_params = 6
        data_size = 1000
        x = np.random.default_rng(1).integers(0, 30, size=num_params * data_size)

        questions, answers = [], []
        for i in range(0, num_params * data_size, num_params):
            a, b, c, d, e, f = x[i:i+6]
            if f == 0 : 
                f = 1
            question = f'What is the result of {a}+{b}*{c}+{d}-{e}÷{f}?'
            answer = a + b * c + d - e / f
            questions.append(question)
            answers.append(str(answer))
        
    elif args.dataset in ['formal_logic', 'pro_med']:
        if args.dataset == 'formal_logic':
            ds = load_dataset('cais/mmlu', 'formal_logic')['test']
        else:
            ds = load_dataset('cais/mmlu', 'professional_medicine')['test']
        ds = pd.DataFrame(ds)
        
        questions = []
        answers = []
        choices = "ABCD"
        for query, options, answer in zip(ds['question'], ds['choices'], ds['answer']):
            if len(options) != 4 :
                continue
            question = '{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n\n'.format(query, options[0], options[1], options[2], options[3])
            label = f"({choices[int(answer)]})"
            questions.append(question)
            answers.append(label)
    
    elif args.dataset in ['crux_eval']:
        ds = load_dataset('cruxeval-org/cruxeval')['test']
        format_code = "Code: {code}\n\nInput: {input}"
        questions = [format_code.format(code=item['code'], input=item['input']) for item in ds]
        answers = [item['output'] for item in ds]
    
    elif args.dataset in ['mmlu_pro']:
        questions, answers = load_mmlu_pro(split='test', n_sample_per_cat=10)
        
    else:
        raise ValueError(f"Dataset {args.dataset} not supported for parsing.")
    

    # Build few-shot prompt
    n_few = min(args.few_shot_num, len(questions))
    if args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med', 'crux_eval', 'mmlu_pro']:
        few_shot_prompt = get_instruction_suffix(args=args)
        
    elif args.dataset == 'coqa':
        few_shot_prompt = f"This is a bot that correctly answers questions based on the provided context.\n\n{stories[0]}\n\n"
        
    elif args.dataset in ['trivia_qa', 'truthful_qa']:
        pass  # already constructed above
    else:
        few_shot_prompt = "This is a bot that correctly answers questions.\n"
        for i in range(n_few):
            if args.dataset in ['sciq', 'nq']:
                few_shot_prompt += f"Question: {questions[i]}\nAnswer: {answers[i]}\n\n"
            elif args.dataset == 'arith':
                few_shot_prompt += f"{questions[i]}\n{answers[i]}\n\n" # Question and Answer is included in context
            else:
                few_shot_prompt += f"Question: {questions[i]}\nAnswer: {answers[i]}\n\n"
    
    # Construct processed dataset
    processed_dataset = []
    if args.dataset == 'coqa':
        for i in range(n_few, len(questions)):
            prompt = few_shot_prompt + questions[i]
            processed_dataset.append({
                "question": questions[i],
                "answer": answers[i],
                "prompt": prompt
            })
    elif args.dataset == 'arith':
        for i in range(n_few, len(questions)):
            prompt = few_shot_prompt + f"{questions[i]}\n"
            processed_dataset.append({
                "question": questions[i],
                "answer": answers[i],
                "prompt": prompt
            })
    elif args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med', 'crux_eval', 'mmlu_pro']:
        for i in range(len(questions)):
            prompt = f"Question: {questions[i]}\n" + few_shot_prompt
            processed_dataset.append({
                "question": questions[i],
                "answer": answers[i],
                "prompt": prompt
            })
    else:
        for i in range(n_few, len(questions)):
            prompt = few_shot_prompt + f"Question: {questions[i]}\nAnswer:"
            processed_dataset.append({
                "question": questions[i],
                "answer": answers[i],
                "prompt": prompt
            })

    return processed_dataset


# --------------------
# GENERATION + NLL
# --------------------
# generate_sequences_hf: Huggingface version for GPT-OSS-20B
def generate_sequences(llm, dataset, rouge, args):
    
    print('--- GENERATION PARAMETERS ---')
    print('Dataset:', args.dataset)
    print('Model:', args.model)
    
    greedy_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0,
        n=1,
        logprobs=1
    )
    multinomial_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=1,
        n=args.n_samples,
        logprobs=1
    )

    sequences = []
    for i, batch in enumerate(tqdm(dataset)):
        prompt = batch['prompt']
        question = batch['question']
        answer = batch['answer'].strip()

        # === GREEDY DECODING ===
        greedy_out = llm.generate(prompt, sampling_params=greedy_params, use_tqdm=False)[0].outputs[0]
        greedy_text_raw = greedy_out.text.strip()
        if args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med', 'mmlu_pro']:
            
            greedy_text = extract_math_response(text=greedy_text_raw, args=args)
            if args.dataset in ['gsm8k']:
                answer = extract_math_answer(answer)
            elif args.dataset in ['arith_long']:
                answer = float(answer)
        
        elif args.dataset in ['crux_eval']:
            
            greedy_text = extract_code_response(greedy_text_raw)
            answer = normalize_answer(answer)

        else:
            
            greedy_text = clean_generation(greedy_text_raw)
            
        greedy_logprobs = greedy_out.logprobs

        llm_label = None
        if args.dataset in ['svamp', 'arith']: # exact match for math datasets
            eval_score = compute_label(greedy_text, answer, eval_method='exact_match')
        
        elif args.dataset in ['gsm8k', 'arith_long']: # exact match after rounding to 1 decimal place for math datasets
            eval_score = int(greedy_text == np.round(answer, 1))
        
        elif args.dataset in ['formal_logic', 'pro_med', 'mmlu_pro']: # exact match for multiple choice datasets
            eval_score = int(greedy_text == answer)
            
        elif args.dataset in ['crux_eval']: # exact match for code generation
            eval_score = code_eval(greedy_text, answer)
            
        else:
            eval_score = compute_label(greedy_text, answer, rouge=rouge, eval_method='rougeL')
            llm_label = compute_label(greedy_text, answer, question=question, eval_method='llm_eval', api_type=args.api_type)

        # === MULTINOMIAL DECODING ===
        sampled_outputs = llm.generate(prompt, sampling_params=multinomial_params, use_tqdm=False)[0].outputs
        generated_texts = [o.text for o in sampled_outputs]
        generation_logprobs = [o.logprobs for o in sampled_outputs]

        # === CLEANING ===
        cleaned = [clean_generation(g) for g in generated_texts]
        if args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med', 'mmlu_pro']:
            extracted_answers = [extract_math_response(text=g, args=args) for g in generated_texts]
        
        elif args.dataset in ['crux_eval']:
            extracted_answers = [extract_code_response(g) for g in generated_texts]
        
        else:
            extracted_answers = [clean_generation(g) for g in generated_texts]

        # === UNCERTAINTY (negative log-likelihood) ===
        samples_avg_nll, samples_nll = [], []
        
        for sample in generation_logprobs:
            flat = flatten_logprobs(sample)
            avg_nll = -np.mean(flat) if len(flat) > 0 else np.nan
            nll = -np.sum(flat) if len(flat) > 0 else np.nan
            samples_avg_nll.append(avg_nll)
            samples_nll.append(nll)

        greedy_flat = flatten_logprobs(greedy_logprobs)
        greedy_avg_nll = -np.mean(greedy_flat) if greedy_flat else np.nan
        greedy_nll = -np.sum(greedy_flat) if greedy_flat else np.nan

        # === DEBUG PRINTS ===
        if i < 5:
            print('Prompt:', prompt)
            print('Question:', question)
            print('Answer:', answer)
            print('Greedy text:', greedy_text_raw)
            print('Cleaned greedy text:', greedy_text)
            print('Greedy logprobs:', greedy_logprobs)
            print('Generated texts:', generated_texts)
            print('Extracted greedy answer:', greedy_text)
            print('Extracted answers:', extracted_answers)
            print('Samples avg NLL:', samples_avg_nll)
            print('Samples NLL:', samples_nll)
            print('Eval score:', eval_score)
            print('---')

        # === STRUCTURE OUTPUT ===
        sequences.append({
            'id': f"{args.dataset}_{i}",
            'prompt': prompt,
            'question': question,
            'answer': answer,
            'generated_texts': generated_texts,
            'cleaned_generated_texts': cleaned,
            'extracted_answers': extracted_answers,
            'samples_nll': samples_nll,
            'samples_avg_nll': samples_avg_nll,
            'greedy_text': greedy_text,
            'greedy_text_raw': greedy_text_raw,
            'greedy_nll': greedy_nll,
            'greedy_avg_nll': greedy_avg_nll,
            'eval_score': eval_score,
            "llm_label": llm_label,
            "max_new_tokens": args.max_new_tokens
        })
        
    return sequences

### ---------------------------------
### Label computation utils
### ---------------------------------
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
        message = message[0]["content"] if message else "" # [role: user, content: prompt]
        response = gemini_client.models.generate_content(model=api_key.gemini_model, contents=message)
        return response.text
    except Exception as e:
        print("Evaluation Error: ", e)
        print(message)
        return None

def compute_label(generation, ground_truth, question=None, rouge=None, eval_method="exact_match", api_type='cohere'):
    if eval_method == "exact_match":
        return generation == ground_truth
    elif eval_method == "rougeL":
        rouge_output = rouge.compute(predictions=[generation], references=[ground_truth])
        return rouge_output["rougeL"]
    elif eval_method == "llm_eval":
        message = [
                {
                    "role": "user", 
                    "content": f"""You are a helpful and precise assistant for checking the quality of the answer.
                    Question: {question}
                    Ground Truth: {ground_truth}
                    Answer To Be Evaluated: {generation}
                    Is the answer correct? Please answer with 'Yes' or 'No'."""}
        
            ]
        try:
            if api_type == 'cohere':
                llm_response = cohere_evaluate(message)
            elif api_type == 'gemini':
                llm_response = gemini_evaluate(message)
            else:
                raise ValueError(f"Unsupported API type: {api_type}")
        except Exception as e:
            print(f"Error during LLM evaluation: {e}")
            llm_response = None
        
        return 1 if llm_response and llm_response.strip().lower().startswith("yes") else 0
    else:
        raise ValueError(f"Unsupported eval method: {eval_method}")


def compute_ece(confidence, labels, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            acc = np.mean(1 - labels[mask])  # 1 = correct
            conf = np.mean(confidence[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(labels)

    return ece

def minmax_normalize(x):
    x = np.array(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

### ---------------------------------
### Metric computation utils
### ---------------------------------
def compute_weighted_mean(embeddings, weights):
    """Compute weighted mean of embeddings."""
    weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(embeddings.device)
    return (weights * embeddings).sum(dim=0)


def compute_metrics(embeddings, mean, weights, p=1):
    """
    Compute various distance-based metrics between embeddings and a mean embedding.
    
    Args:
        embeddings (torch.Tensor): shape (N, D)
        mean (torch.Tensor): shape (D,)
        weights (array-like): length N, non-negative
        p (float): order for Lp or Wasserstein metric
        method (str): one of {"lp", "wasserstein", "cosine", "mahalanobis"}
        cov_inv (torch.Tensor, optional): inverse covariance for Mahalanobis distance
    """
    device = embeddings.device
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    weights_tensor = weights_tensor / weights_tensor.sum()  # normalize weights
    diffs = embeddings - mean

    norms = torch.norm(diffs, p=p, dim=1)
    weighted = (weights_tensor * norms).sum()
    return weighted.item()



### --------------------------------- Self-certainty ------------------------------------------
def confidence_logprob_sum(logprob_sum: torch.Tensor, attention_mask: torch.Tensor, V: int):
    """
    Calculate the confidence of the logprob_sum.
    logprob_sum: torch.Tensor, shape (batch_size, seq_length) or (seq_length)
    attention_mask: torch.Tensor, shape (batch_size, seq_length) or (seq_length)
    V: int, the vocab size
    """
    logprob_sum = logprob_sum.contiguous()
    attention_mask = attention_mask.contiguous()
    V_tensor = torch.tensor(V, dtype=logprob_sum.dtype, device=logprob_sum.device)
    conf = -1/V * logprob_sum - torch.log(V_tensor)
    valid_conf = conf * attention_mask
    batch_confidence_list = (valid_conf.sum(dim=-1) / attention_mask.sum(dim=-1)).tolist()
    return batch_confidence_list

def get_self_certainty_sample(all_confidences, answers, power=0.3):
    sorted_indices = sorted(range(len(all_confidences)), key=lambda k: all_confidences[k], reverse=True)
    votes_per_output = [len(all_confidences) - rank for rank in range(len(all_confidences))] 

    # Power function votes
    votes_per_output = [vote**power for vote in votes_per_output]


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
            
    all_confidences = [votes[i] for i in range(len(all_confidences))]

    best_confidence = max(all_confidences)
    best_index = all_confidences.index(best_confidence)
    return answers[best_index]


@torch.no_grad()
def compute_self_certainty_scores(
    model_dir: str,
    prompts: list[str],
    generated_texts_list: list[list[str]], 
    batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_length: int = 2048,
) -> list[list[float]]:
    # Reference from: https://github.com/backprop07/Self-Certainty/blob/main/src/confidence_list.py
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None
    ).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    all_confidences = []

    for idx, (prompt, generated_texts) in enumerate(tqdm(zip(prompts, generated_texts_list), total=len(prompts))):
        # Encode prompt
        prompt_enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        ).to(device)
        input_ids = prompt_enc.input_ids[0]
        input_mask = prompt_enc.attention_mask[0]
        input_len = input_mask.sum().item()

        confidences = [None] * len(generated_texts)

        # Group generated texts by length to avoid OOM
        groups = {"small": [], "medium": [], "large": []}
        indices = []
        for i, text in enumerate(generated_texts):
            l = len(text)
            if l > 6144:
                groups["large"].append(text)
            elif l > 3072:
                groups["medium"].append(text)
            else:
                groups["small"].append(text)
            indices.append(i)

        group_bs = {"small": batch_size, "medium": max(1, batch_size//2), "large": max(1, batch_size//4)}

        for group_name in ["small", "medium", "large"]:
            texts = groups[group_name]
            if not texts:
                continue

            group_indices = [indices[i] for i in range(len(indices)) if generated_texts[indices[i]] in texts]
            bs = group_bs[group_name]

            # Tokenize outputs
            out_enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            out_ids = out_enc.input_ids
            out_mask = out_enc.attention_mask

            # Repeat prompt for batch
            full_ids = torch.cat([
                input_ids.unsqueeze(0).repeat(len(texts), 1),
                out_ids
            ], dim=1).long()
            full_mask = torch.cat([
                input_mask.unsqueeze(0).repeat(len(texts), 1),
                out_mask
            ], dim=1).long()

            group_confs = []
            for i in range(0, len(texts), bs):
                j = i + bs
                batch_ids = full_ids[i:j]
                batch_mask = full_mask[i:j]

                logits = model(batch_ids, attention_mask=batch_mask).logits
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    batch_logprob_sum = logits[:, input_len:, :] 
                    batch_logprob_sum = F.log_softmax(batch_logprob_sum, dim=-1)
                    batch_logprob_sum = batch_logprob_sum.sum(dim=-1).to(device).to(torch.float32)
                
                # Use the output attention mask from the tokenized group (for this batch).
                batch_output_attention_mask = out_mask[i:j]
                
                vocab_size = getattr(model.config, "vocab_size", None)
                if vocab_size is None:
                    vocab_size = model.get_input_embeddings().weight.shape[0]
                batch_confidence_list = confidence_logprob_sum(batch_logprob_sum, batch_output_attention_mask, vocab_size) # model.config.vocab_size
                group_confs.extend(batch_confidence_list)

            for conf, orig_idx in zip(group_confs, group_indices):
                confidences[orig_idx] = float(conf)

        all_confidences.append(confidences)

    return all_confidences


# ------------------------- ModeX -----------------------------
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

    # === 1. adjacency matrix ===
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

    # === 2. recursive spectral graph cut ===
    _A = A.copy()
    current_names = agent_names.copy()

    while True:
        # Spectral clustering (2-way cut)
        info = graph_cut(_A, current_names) 
        # iter_count += 1
        # if iter_count > max_iter:
        #     break
        
        # Choose bigger group
        g1 = info['groups']['group_1']
        g2 = info['groups']['group_2']
        group = g1 if len(g1) >= len(g2) else g2
        n_group = g2 if len(g1) >= len(g2) else g1

        prev_n = len(current_names)
        group_indices = [current_names.index(name) for name in group]
        n_group_indices = [current_names.index(name) for name in n_group]

        # Goodness of cut
        phi = goodness_of_cut_func(_A, group_indices, n_group_indices, goodness_of_cut)

        # Highest degree node
        if phi >= tau:
            degrees = np.sum(_A, axis=1)
            best_idx = int(np.argmax(degrees))
            best_name = current_names[best_idx]
            return agent_names.index(best_name)   # trả về index gốc

        _A = _A[np.ix_(group_indices, group_indices)]
        current_names = [current_names[i] for i in group_indices]

        if len(current_names) == prev_n or len(current_names) <= 1:
            break

    # Fallback: random
    selected_name = np.random.choice(current_names)
    return agent_names.index(selected_name)


# ====================== UTILITY ======================

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

    # CACHE ONCE
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

    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]  

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
    else:  
        vol_s = float(np.sum(A[group_indices, :]) - len(group_indices))
        vol_sbar = float(np.sum(A[n_group_indices, :]) - len(n_group_indices))
        return (cut / vol_s) + (cut / vol_sbar)


### --------------------------------- GSM8K ------------------------------------------
def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def extract_math_answer(full_ans_text: str) -> str:
    match = ANS_RE.search(full_ans_text)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return int(match_str.strip())
    return None

# ------------------------------
def get_instruction_suffix(args):
    if args.dataset in ['arithmetics', 'arith_long']:
        return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 12.34}'. Let's think step by step."
    elif args.dataset in ['gsm8k']:
        return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 123}'. Let's think step by step."
        
    elif args.dataset in ['hellaswag','pro_med','formal_logic','csqa','hh_rlhf', 'mmlu_pro']:
        return " Make sure to state your final answer choice in curly brackets at the very end of your response, just like: '{final answer: (A)}'. Let's think step by step."
    
    elif args.dataset in ['cnn_daily']:
        return ' Make sure to provide your summary after stating "# Summary # ".'
    
    elif args.dataset in ['crux_eval']:
        prompt = """You are given a Python function and some inputs. 
Your task is to determine the exact output of the function when called with those inputs.

Think step by step inside your mind, but **DO NOT output any reasoning, explanation, or extra text**.
Your response MUST be **ONLY** a valid JSON object in this exact format:
{"answer": <final_output>}
Do not include any markdown, code blocks, or additional text outside the JSON."""
        
        return prompt
    
### ---------------------------------- Arithmetics ------------------------------------------
def load_data(args, split=None, easy=False):
    
    data_size = args.data_size
    num_params = 4 if easy else 6

    if split == 'train' :
        x = np.random.default_rng(0).integers(0, 30, size=num_params * data_size)
    else :
        x = np.random.default_rng(1).integers(0, 30, size=num_params * data_size)

    X, Y = [], []
    for i in range(0, num_params * data_size, num_params):
        if easy :
            a, b, c, d = x[i:i+4]
            question = f'What is the result of {a}+{b}*{c}-{d}?'
            answer = a + b * c - d
        else :
            a, b, c, d, e, f = x[i:i+6]
            if f == 0 : 
                f = 1
            question = f'What is the result of {a}+{b}*{c}+{d}-{e}÷{f}?'
            answer = a + b * c + d - e / f
        X.append(question)
        Y.append(answer)
    
    return X, Y