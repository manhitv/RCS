from http.client import responses

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import re
import random
import os
import config
import api_key
import cohere
from google import genai
import networkx as nx
from scipy.linalg import expm


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
    "mistral-large": "mistralai/Mistral-Large-Instruct-2407"
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


def extract_math_response(text, args):
    try:
        pred = re.findall(r"\{(.*?)\}", text)[-1]
        pred = pred.replace("final answer:", "").strip()
        
        if args.dataset in ['gsm8k', 'arith_long']:
            pred = float(pred)
            text = round(pred, 1)
        
        elif args.dataset in ['formal_logic', 'pro_med']:
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
                batch_confidence_list = confidence_logprob_sum(batch_logprob_sum, batch_output_attention_mask, model.config.vocab_size)
                group_confs.extend(batch_confidence_list)

            for conf, orig_idx in zip(group_confs, group_indices):
                confidences[orig_idx] = float(conf)

        all_confidences.append(confidences)

    return all_confidences

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

def get_instruction_suffix(args):
    if args.dataset in ['arithmetics', 'arith_long']:
        return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 12.34}'. Let's think step by step."
    elif args.dataset in ['gsm8k']:
        return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 123}'. Let's think step by step."
        
    elif args.dataset in ['hellaswag','pro_med','formal_logic','csqa','hh_rlhf']:
        return " Make sure to state your final answer choice in curly brackets at the very end of your response, just like: '{final answer: (A)}'. Let's think step by step."
    
    elif args.dataset in ['cnn_daily']:
        return ' Make sure to provide your summary after stating "# Summary # ".'


def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append("There are 15 trees in the grove. "
                    "Grove workers will plant trees in the grove today. "
                    "After they are done, there will be 21 trees. "
                    "How many trees did the grove workers plant today?")
    chain.append("There are 15 trees originally. "
                 "Then there were 21 trees after some more were planted. "
                 "So there must have been 21 - 15 = 6.")
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?")
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?")
    chain.append("Originally, Leah had 32 chocolates. "
                 "Her sister had 42. So in total they had 32 + 42 = 74. "
                 "After eating 35, they had 74 - 35 = 39.")
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?")
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8.")
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?")
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9.")
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?")
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29.")
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?")
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls.")
    answer.append("33")

    question.append("Olivia has $23. She bought five bagels for $3 each. "
                    "How much money does she have left?")
    chain.append("Olivia had 23 dollars. "
                 "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
                 "So she has 23 - 15 dollars left. 23 - 15 is 8.")
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Question: " + question[i] + "\nAnswer: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text

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

### --------------------------------- EigenEmbed ------------------------------------------
def compute_eigen_embed(sentence_embeddings, alpha=1e-3):
    embeddings = (
        sentence_embeddings.cpu().numpy() 
        if isinstance(sentence_embeddings, torch.Tensor) 
        else np.array(sentence_embeddings)
    )
    if embeddings.ndim != 2:
        raise ValueError(f"Expected (N, D) tensor, got {embeddings.shape}")
    
    embeddings = np.array(embeddings)  # Shape: (N, D)
    N = embeddings.shape[0]
    
    # CRUCIAL: Use sample covariance (N x N), NOT feature covariance
    cov = np.cov(embeddings) # Shape: (N, N)
    cov = cov + alpha * np.eye(N)  # Regularize

    # SVD
    u, s, vT = np.linalg.svd(cov)  # singular values
    s = np.sort(s)[::-1]  # descending order
    eigen_score = np.mean(np.log10(s))

    return eigen_score


# ---------------- 
### PRO Score
# ----------------
def approx(probs):
    """Compute PRO score from probabilities."""
    pk = probs[-1]
    score = -np.log(pk) - np.sum([pi * np.log(pi / pk) for pi in probs[:-1]])
    return score


def pro_score(generation, alpha=0.4):
    """Compute PRO score from generation."""
    nll_probs = np.exp(-np.array(generation["samples_nll"]))
    top_probs = np.sort(nll_probs)[::-1]
    filtered = top_probs[top_probs >= alpha]
    if len(filtered) == 0:
        filtered = np.array([top_probs[0]])
        
    return approx(filtered)

# -------------------------------------
### Semantic Entropy
# -------------------------------------
# Compute semantic similarity set IDs
def compute_semantic_similarity(sample, semantic_model, semantic_tokenizer, device='cuda:0'):

    question = sample['question']
    generations = sample['cleaned_generated_texts']
    unique_generations = list(set(generations))
    semantic_set_ids = {ans: i for i, ans in enumerate(unique_generations)}

    # pairwise DeBERTa similarity
    for i, a1 in enumerate(unique_generations):
        for j, a2 in enumerate(unique_generations):
            if j <= i:
                continue
            qa1 = question + " " + a1
            qa2 = question + " " + a2

            # NLI prediction: 0 = contradiction, 1 = neutral, 2 = entailment
            encoded = semantic_tokenizer(qa1, qa2, return_tensors='pt', truncation=True, max_length=512).to(device)
            logits = semantic_model(**encoded).logits
            pred = torch.argmax(logits, dim=1).item()

            encoded_rev = semantic_tokenizer(qa2, qa1, return_tensors='pt', truncation=True, max_length=512).to(device)
            logits_rev = semantic_model(**encoded_rev).logits
            pred_rev = torch.argmax(logits_rev, dim=1).item()

            if not(pred == 0 or pred_rev == 0):
                semantic_set_ids[a2] = semantic_set_ids[a1]

    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generations]

    return list_of_semantic_set_ids

# Main Semantic Entropy Computation
def compute_semantic_entropy(sample, embed_model, embed_tokenizer, device='cuda:0'):
    """Compute semantic entropy for a single sample.
    Args:
        sample (dict): Dictionary containing 'samples_avg_nll' and 'cleaned_generated_texts'.
        embed_model: Transformer model for embedding and prediction.
        embed_tokenizer: Tokenizer for encoding inputs.
    """
    # Semantic set
    semantic_set_ids = compute_semantic_similarity(sample, embed_model, embed_tokenizer, device)
    
    # Convert inputs to tensors
    avg_nll = torch.as_tensor(sample['samples_avg_nll'], dtype=torch.float32)
    log_probs = -avg_nll  # Convert NLL to log-probabilities -- important for correct entropy calculation
    semantic_set_ids = torch.as_tensor(semantic_set_ids, dtype=torch.int64)

    # Get unique semantic set IDs (excluding -1)
    valid_set_ids = torch.unique(semantic_set_ids[semantic_set_ids != -1])
    
    if valid_set_ids.numel() == 0:
        return [0, 0, 0]  # No valid sets, return zero entropy
    
    # Aggregate log-likelihoods for each semantic set
    aggregated_log_probs, dse_log_probs = [], []
    for set_id in valid_set_ids:
        mask = (semantic_set_ids == set_id)
        agg_log_prob = torch.logsumexp(log_probs[mask], dim=0)
        aggregated_log_probs.append(agg_log_prob)
        dse_log_probs.append(len(mask.nonzero()) / len(semantic_set_ids))  # DSE weight based on set size
    
    # Convert to tensor and compute probabilities
    # aggregated_log_probs = torch.tensor(aggregated_log_probs, dtype=torch.float32, device=avg_nll.device)
    aggregated_log_probs = torch.stack(aggregated_log_probs)

    probs = torch.softmax(aggregated_log_probs, dim=0)  # Normalize to probabilities
    
    dse_log_probs = torch.tensor(dse_log_probs, dtype=torch.float32, device=aggregated_log_probs.device)
    dse_probs = torch.softmax(dse_log_probs, dim=0)  # Normalize DSE weights to probabilities
    
    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=0)  # Add epsilon to avoid log(0)
    dse_entropy = -torch.sum(dse_probs * torch.log(dse_probs + 1e-10), dim=0)
    
    return entropy.item(), dse_entropy.item(), len(valid_set_ids)

# -------------------------------------
### Compute DEg and Semantic Density
# -------------------------------------
def compute_deg_semantic_density(sample, embed_model, embed_tokenizer, device='cuda:0'):
    """Compute contradiction probability (deg) and semantic density for a single sample as scalars.
    
    Args:
        sample (dict): Dictionary containing 'question', 'cleaned_generated_texts', 
                       'greedy_text', and 'samples_avg_nll'.
        embed_model: Transformer model for embedding and prediction.
        embed_tokenizer: Tokenizer for encoding inputs.
        device (str): Device to run computations on (default: 'cuda:0').
    
    Returns:
        tuple: (deg, sd), where deg and sd are Python scalars (float).
    """
    question = sample['question']
    cleaned_generated_texts = sample['cleaned_generated_texts']
    most_likely_text = str(sample['greedy_text'])
    contradict_prob_list = []

    likelihood_sum = 0.0
    semantic_density = 0.0
    
    # Evaluate semantic similarity
    unique_cleaned_generation = set()
    unique_index = []

    for generation_index in range(len(cleaned_generated_texts)):
        generation_text = cleaned_generated_texts[generation_index]
        if generation_text not in unique_cleaned_generation:
            unique_cleaned_generation.add(generation_text)
            unique_index.append(generation_index)

    # Semantic Density & Deg matrix
    for generation_index in unique_index:
        qa_1 = question + ' ' + cleaned_generated_texts[generation_index]
        qa_2 = question + ' ' + most_likely_text
        average_likelihood = float(np.exp(-sample['samples_avg_nll'][generation_index]))
        origin_input = qa_1 + ' [SEP] ' + qa_2

        # Encode and predict for forward input
        encoded_input = embed_tokenizer(origin_input, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            prediction = embed_model(**encoded_input).logits[0]
        
        # Apply torch.softmax and convert to scalars
        prediction_softmax = torch.softmax(prediction, dim=-1)
        contradict_prob_1 = float(1 - prediction_softmax[2].item())
        semantic_distance = float(prediction_softmax[0].item() + 0.5 * prediction_softmax[1].item())
        semantic_density += 0.5 * (1.0 - semantic_distance) * average_likelihood

        # Encode and predict for reverse input
        reverse_input = qa_2 + ' [SEP] ' + qa_1
        encoded_reverse_input = embed_tokenizer(reverse_input, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            reverse_prediction = embed_model(**encoded_reverse_input).logits[0]
        
        # Apply torch.softmax and convert to scalars
        reverse_prediction_softmax = torch.softmax(reverse_prediction, dim=-1)
        contradict_prob_2 = float(1 - reverse_prediction_softmax[2].item())
        reverse_semantic_distance = float(reverse_prediction_softmax[0].item() + 0.5 * reverse_prediction_softmax[1].item())
        
        # Update metrics
        semantic_density += 0.5 * (1.0 - reverse_semantic_distance) * average_likelihood
        likelihood_sum += average_likelihood
        contradict_prob_list.append((contradict_prob_1 + contradict_prob_2) / 2.0)
    
    # Compute final metrics as scalars
    deg = np.mean(contradict_prob_list) if contradict_prob_list else 0.0
    sd = 1 - semantic_density / likelihood_sum if likelihood_sum > 0 else 1

    return deg, sd


import numpy as np
import torch
import networkx as nx
from scipy.linalg import expm

def compute_graph_baselines(
    generated_texts,
    semantic_model,
    semantic_tokenizer,
    device="cuda:0",
    temp=1.0,
    kle_t=1.0,      # heat kernel temperature
    eps=1e-12
):
    """
    Graph baselines + KLE (Kernel Language Entropy)

    Returns:
        dict:
            LexicalSim
            Deg
            Ecc
            EigenLap
            KLE
    """

    N = len(generated_texts)
    num_labels = semantic_model.config.num_labels
    logits_mat = np.zeros((N, N, num_labels))

    # ---------------------------
    # Step 1 — pairwise NLI logits
    # ---------------------------
    for i in range(N):
        for j in range(N):
            input_text = generated_texts[i] + " [SEP] " + generated_texts[j]

            encoded = semantic_tokenizer(
                input_text,
                padding=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                logits = semantic_model(**encoded).logits[0].cpu().numpy()

            logits_mat[i, j] = logits

    # ---------------------------
    # Step 2 — entailment adjacency
    # ---------------------------
    probs = np.exp(logits_mat / temp)
    probs /= probs.sum(-1, keepdims=True)

    adjacency = probs[..., 0]  # entailment
    adjacency = 0.5 * (adjacency + adjacency.T)  # make symmetric
    np.fill_diagonal(adjacency, 0)

    # ---------------------------
    # Baseline 1 — Lexical similarity proxy
    # ---------------------------
    lexical_sim = adjacency.mean()

    # ---------------------------
    # Degree & UDeg
    # ---------------------------
    degree_vals = adjacency.sum(axis=1)
    degree_matrix = np.diag(degree_vals)

    m = adjacency.shape[0]
    Udeg = np.trace(m * np.eye(m) - degree_matrix) / (m**2)

    # ---------------------------
    # Laplacian
    # ---------------------------
    laplacian = degree_matrix - adjacency

    eigvals_L = np.linalg.eigvalsh(laplacian)
    sum_eigenvalues = np.sum(eigvals_L)

    # ---------------------------
    # Eccentricity
    # ---------------------------
    G = nx.from_numpy_array(adjacency)

    for u, v, d in G.edges(data=True):
        d["weight"] = 1.0 / (d["weight"] + 1e-8)

    ecc = nx.eccentricity(
        G,
        sp=dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    )

    avg_eccentricity = np.mean(list(ecc.values()))

    # =========================================================
    # 🧠 KLE — Kernel Language Entropy
    # =========================================================

    # Heat kernel from Laplacian
    K = expm(-kle_t * laplacian)

    # unit trace normalization
    K /= np.trace(K)

    eigvals_K = np.linalg.eigvalsh(K)
    eigvals_K = np.clip(eigvals_K, eps, None)

    KLE = -np.sum(eigvals_K * np.log(eigvals_K))

    # ---------------------------
    return {
        "LexicalSim": lexical_sim,
        "Deg": Udeg,
        "Ecc": avg_eccentricity,
        "EigenLap": sum_eigenvalues,
        "KLE": KLE,
    }
    

### Semantic Volume
def semantic_volume(
    embeddings,
    pca_dim=10,
    eps=1e-12,
    log_volume=True
):
    """
    embeddings: (N, D) tensor
    """

    X = embeddings
    N, D = X.shape

    if N <= 1:
        return torch.tensor(0.0, device=X.device)

    # center
    X = X - X.mean(dim=0, keepdim=True)

    # PCA via SVD
    k = min(pca_dim, N, D)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)

    # project
    X_pca = X @ Vh[:k].T

    cov = (X_pca.T @ X_pca) / (N - 1)

    eigvals = torch.linalg.eigvalsh(cov).clamp_min(eps)

    if log_volume:
        return 0.5 * torch.sum(torch.log(eigvals))
    else:
        return torch.sqrt(torch.prod(eigvals))

 
### Compute P(True)
def get_logprob_of_token(
    model,
    tokenizer,
    prompt: str,
    target_token: str = "A",
    device: str = "cuda"
) -> float:
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab)

    # next-token distribution
    next_token_logits = logits[0, -1]  # (vocab,)

    log_probs = F.log_softmax(next_token_logits, dim=-1)

    target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]

    return log_probs[target_id].item()

def calculate_p_true(
    model,
    tokenizer,
    question: str,
    most_probable_answer: str,
    brainstormed_answers: list[str],
    few_shot_prompt: str = "",
    hint: bool = False,
    device: str = "cuda",
) -> float:

    # ===== Build prompt =====
    prompt = f"{few_shot_prompt}\n" if few_shot_prompt else ""

    prompt += f"Question: {question}\n"

    prompt += "Brainstormed Answers:\n"
    for ans in brainstormed_answers:
        prompt += f"{ans.strip()}\n"

    prompt += f"{most_probable_answer.strip()}\n"
    prompt += f"Possible answer: {most_probable_answer}\n"

    if not hint:
        prompt += (
            "Is the possible answer:\n"
            "A) True\n"
            "B) False\n"
            "The possible answer is:"
        )
    else:
        prompt += (
            "Do the brainstormed answers match the possible answer? "
            "Respond with A if they do, B if they do not. Answer:"
        )

    # ===== Compute log P(True) =====
    log_p_true = get_logprob_of_token(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_token="A",
        device=device
    )

    return log_p_true