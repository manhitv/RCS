import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import pickle
import numpy as np
import pandas as pd
import tqdm
import evaluate
import config
import json
import re
import logging
logging.basicConfig(level=logging.ERROR)

from datasets import load_dataset
from vllm import LLM, SamplingParams
from utils import (
    MODEL_PATH_DICT,
    clean_answer, 
    is_correct, 
    create_demo_text, 
    set_seed, 
    clean_generation, 
    flatten_logprobs,
    compute_label,
    get_instruction_suffix,
    extract_math_answer,
    extract_math_response
    )

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
    
    else:
        raise ValueError(f"Dataset {args.dataset} not supported for parsing.")

    # Build few-shot prompt
    n_few = min(args.few_shot_num, len(questions))
    if args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med']:
        few_shot_prompt = get_instruction_suffix(args=args) # create_demo_text()
        
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
    elif args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med']: 
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
    for i, batch in enumerate(tqdm.tqdm(dataset)):
        prompt = batch['prompt']
        question = batch['question']
        answer = batch['answer'].strip()

        # === GREEDY DECODING ===
        greedy_out = llm.generate(prompt, sampling_params=greedy_params, use_tqdm=False)[0].outputs[0]
        greedy_text_raw = greedy_out.text.strip()
        if args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med']:
            
            greedy_text = extract_math_response(text=greedy_text_raw, args=args)
            if args.dataset in ['gsm8k']:
                answer = extract_math_answer(answer)
            elif args.dataset in ['arith_long']:
                answer = float(answer)

        else:
            
            greedy_text = clean_generation(greedy_text_raw)
            
        greedy_logprobs = greedy_out.logprobs

        llm_label = None
        if args.dataset in ['svamp', 'arith']: # exact match for math datasets
            eval_score = compute_label(greedy_text, answer, eval_method='exact_match')
        
        elif args.dataset in ['gsm8k', 'arith_long']: # exact match after rounding to 1 decimal place for math datasets
            # model_answer = clean_answer(greedy_text)
            # eval_score = is_correct(model_answer=model_answer, answer=answer)
            eval_score = int(greedy_text == np.round(answer, 1))
        
        elif args.dataset in ['formal_logic', 'pro_med']: # exact match for multiple choice datasets
            eval_score = int(greedy_text == answer)
            
        else:
            eval_score = compute_label(greedy_text, answer, rouge=rouge, eval_method='rougeL')
            llm_label = compute_label(greedy_text, answer, question=question, eval_method='llm_eval', api_type=args.api_type)

        # === MULTINOMIAL DECODING ===
        sampled_outputs = llm.generate(prompt, sampling_params=multinomial_params, use_tqdm=False)[0].outputs
        generated_texts = [o.text for o in sampled_outputs]
        generation_logprobs = [o.logprobs for o in sampled_outputs]

        # === CLEANING ===
        cleaned = [clean_generation(g) for g in generated_texts]
        
        if args.dataset in ['gsm8k', 'formal_logic', 'arith_long', 'pro_med']:
            extracted_answers = [extract_math_response(text=g, args=args) for g in generated_texts]
        else:
            extracted_answers = cleaned

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

# --------------------
# MAIN PIPELINE
# --------------------
def main(args):
    experiment_id = os.getpid()
    cache_dir = f"/tmp/rouge_cache_{experiment_id}"
    os.environ['HF_EVALUATE_CACHE'] = cache_dir
    rouge = evaluate.load('rouge', experiment_id=experiment_id, cache_dir=cache_dir)
    
    # Load dataset
    dataset = parse_dataset(args=args)

    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset[: int(len(dataset) * args.fraction_of_data_to_use)]

    # Init model
    hf_model_dir = MODEL_PATH_DICT[args.model]
    llm = LLM(model=hf_model_dir, dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=2048)
    print(f"Loaded model {args.model} for generation.")
    
    # Check if output already exists
    output_path = f"{config.output_dir}/{args.dataset}_{args.model}_N={args.n_samples}_F={args.fraction_of_data_to_use}_A={args.api_type}_S={args.seed}__generation.pkl"
    # if os.path.exists(output_path):
    #     print(f"Output already exists at {output_path}. Ignore generation results...")
    #     with open(output_path, "rb") as f:
    #         sequences = pickle.load(f)
    #     return sequences
    
    # Run generation
    sequences = generate_sequences(llm=llm, dataset=dataset, rouge=rouge, args=args)
    
    # Save
    # output_path = f"{config.output_dir}/{args.dataset}_{args.model}_N={args.n_samples}_F={args.fraction_of_data_to_use}_A={args.api_type}_S={args.seed}__generation.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(sequences, f)

    print(f"Saved results to {output_path}")
    return sequences

# --------------------
# CLI ENTRY
# --------------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
    parser.add_argument('--few_shot_num', type=int, default=5)
    parser.add_argument('--model', type=str, default='gemma-7b', required=True)
    parser.add_argument('--dataset', type=str, default='coqa', required=True)
    parser.add_argument('--max_new_tokens', type=int, default=32) # 512 for gsm8k
    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
    parser.add_argument('--api_type', type=str, default='cohere', choices=['gemini', 'cohere'], help='API type for LLM evaluation')
    
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)