import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from datautils import map_tensors

@torch.no_grad()
def llm_eval_ppl(model, test_loader, device):
    """
    Evaluate the perplexity of a model on a dataset.

    Args:
        model (Huggging Face LLM): LLM Model
        test_data (torch.utils.data.DataLoader): PyTorch DataLoader.
        device (torch.device): data device.

    Returns:
        float: Perplexity of the model on the dataset.
    """
    nsamples = len(test_loader)
    iterable_test_loader = iter(test_loader)

    nlls = []
    for batch in tqdm(iterable_test_loader):
        batch = map_tensors(batch, device)
        logits = model(**batch).logits
        
        # shift outputs and labels autoregressively.
        logits = logits[:, :-1, :].contiguous()
        # shift_labels = batch["input_ids"][:, 1:].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), shift_labels.view(-1)).float()
        nlls.append(loss)

    # ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    nlls = torch.stack(nlls)
    mask = torch.isnan(nlls)
    nlls = nlls[~mask]
    ppl = np.exp(nlls.mean().item())
    return ppl.item()

@torch.no_grad()
def llm_eval_ppl_v2(model, test_loader, device):
    """
    Evaluate the perplexity of a model on a dataset.

    Args:
        model (Huggging Face LLM): LLM Model
        test_data (torch.utils.data.DataLoader): PyTorch DataLoader.
        device (torch.device): data device.

    Returns:
        float: Perplexity of the model on the dataset.
    """
    nsamples = len(test_loader)
    iterable_test_loader = iter(test_loader)

    nlls = []
    for batch in tqdm(iterable_test_loader):
        inp = torch.squeeze(batch['input_ids'], 1).to(device)
        target = torch.squeeze(batch['input_ids'][:, 1:], 1).to(device)
        causal_attention_mask = torch.squeeze(batch['attention_mask'], 1).to(device)

        out = model(inp)[0]
        out = out[:, :-1, :].contiguous()
        
        # shift outputs and labels autoregressively.
        loss = nn.CrossEntropyLoss()(out.view(-1, out.size(-1)), target.view(-1))

        # neg_log_likelihood = loss.float() * model.seqlen
        neg_log_likelihood = loss.float()
        nlls.append(neg_log_likelihood)

    # ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    nlls = torch.stack(nlls)
    mask = torch.isnan(nlls)
    nlls = nlls[~mask]
    ppl = np.exp(nlls.mean().item())
    return ppl.item()


######################### human_eval benchmark #########################
##### copy from https://github.com/abacaj/code-eval #####
##### first install https://github.com/openai/human-eval.git #####

from human_eval.data import write_jsonl, read_problems
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm
import itertools
import typing

BatchGenerator = typing.Callable[
    [PreTrainedModel, PreTrainedTokenizer, str, int], list[str]
]


# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")


def split_batch(samples: list[str], size=4):
    mini_batches = []

    for i in range(0, len(samples), size):
        mini_batches.append(samples[i : i + size])

    return mini_batches


def run_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples_per_task: int,
    out_path: str,
    generate_batch_completion: BatchGenerator,
    format_tabs: bool = False,
):
    problems = read_problems()
    # problems = dict(itertools.islice(problems.items(), 20))
    samples = []
    pbar = tqdm(total=len(problems) * num_samples_per_task)

    for task_id in problems:
        if format_tabs:
            prompt = problems[task_id]["prompt"].replace("    ", "\t")
        else:
            prompt = problems[task_id]["prompt"]

        batch_completions = generate_batch_completion(
            model, tokenizer, prompt, num_samples_per_task
        )

        for sample in batch_completions:
            result = dict(
                task_id=task_id,
                completion=sample,
            )

            samples += [result]

        pbar.update(num_samples_per_task)

    write_jsonl(out_path, samples)

@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # llama model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    return [filter_code(fix_indents(completion)) for completion in batch_completions]


## copy from repo human-eval/human_eval/evaluate_functional_correctness.py
from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness

# def entry_point(
def eval_funct_coorectness(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)
    return results

if __name__ == "__main__":
    from transformers import LlamaTokenizer, LlamaForCausalLM
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # import pdb
    model_path = "/data2/zujingliu/workspace/LLM/llama/meta-llama/Llama-2-7b-hf"

    tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlamaForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            # cache_dir=cache_dir, 
            # low_cpu_mem_usage=True, 
            device_map="auto"
        )
    
    num_samples_per_task = 10
    out_path = "results/llama/eval.jsonl"
    os.makedirs("results/llama", exist_ok=True)

    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
    )