import time
import argparse
import math
import random
import copy
import os
import sys
import json
import gc
import accelerate
from tqdm import tqdm
import json
from types import SimpleNamespace

import torch
import torch.nn as nn

from logger import get_logger, date_uid
from datautils import get_data_module, map_tensors
from modelutils import *
from pgpruning import *
from evalutils import llm_eval_ppl
import fnmatch
from peft import PeftModel



@torch.no_grad()
def eval_ppl(pruner, test_loader, device, logger, args, mask_type="hard_mask"):
    assert mask_type in ['none_mask', 'soft_mask', 'hard_mask', 'init_mask'], "evaluation mask type should be in ['none_mask', 'soft_mask', 'hard_mask', 'init_mask']"
    
    logger.info("Evaluating mask type: {}".format(mask_type))

    pruner.set_mode('test')
    pruner.set_model_mask_type(mask_type)

    ppl = llm_eval_ppl(pruner.subnet, test_loader, device)
    logger.info('Perplexity: {:.3f}'.format(ppl))

    pruner.clear_model_mask()

    return ppl

def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","piqa","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, batch_size=None):
    """
    This version is for lm_eval==0.4.5, 
    'pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@v0.4.5'
    Not usable
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval import tasks
    from lm_eval import utils as lm_eval_utils
    # from lm_eval.api.registry import ALL_TASKS
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, trust_remote_code=True)
    
    task_manager = tasks.TaskManager()
    if task_list is None:
        task_names = task_manager.all_tasks
    else:
        task_names = task_manager.match_tasks(task_list)
    print("task_names: ", task_names)
        
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    
    # export HF_DATASETS_TRUST_REMOTE_CODE=1
    # os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    # model_args = f"pretrained={model},cache_dir=./llm_weights,use_accelerate={use_accelerate},trust_remote_code=True"
    results = lm_eval.simple_evaluate(
        model=hflm,
        # model="hf",
        # model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
    )['results']
    return results


parser = argparse.ArgumentParser()

parser.add_argument('--exp-dir', type=str, default=None, help='Experiment path that the pruned model saved.')
parser.add_argument("--max-seq-length", default=128, type=int, help="dataset size (default: 128)")
parser.add_argument("--global-hard-mask", action="store_true", help="Use global hard mask for pruning.")
parser.add_argument("--pruned-checkpoint", type=int, default=None, help="The checkpoint of the pruned model.")
parser.add_argument(
        '--tasks',
        nargs='+',
        # default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"],
        default=["boolq", "piqa","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"],
    )
parser.add_argument('--num-shot', type=int, default=0, help="few shot evaluation by lm_eval")

parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation.")
parser.add_argument('--dtype', type=str, default='bf16', choices=['fp32', 'fp16','bf16'], help="Data type for the model.")

eval_args = parser.parse_args()

exp_dir = eval_args.exp_dir
if exp_dir is None:
    raise ValueError("exp_dir is not set.")

if not os.path.exists(exp_dir):
    raise ValueError("exp_dir does not exist: {}".format(exp_dir))


# create a subfile in exp_dir to save the eval results
eval_dir = os.path.join(exp_dir, "eval")
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

# logger
logger = get_logger(checkpoint_path=eval_dir, log_filename='eval_{}.log'.format(date_uid()))

logger.info("Evaluate the pruned model.")


parms = json.load(open(os.path.join(exp_dir, "params.json"), "r"))
args = SimpleNamespace(**parms)
args.global_hard_mask = parms.get('global_hard_mask', None)
if args.global_hard_mask is None:
    args.global_hard_mask = True if args.init_type in ["wanda-sp", "random"] else False
else:
    if args.global_hard_mask == "True":
        args.global_hard_mask = True
    elif args.global_hard_mask == "False":
        args.global_hard_mask = False
    else:
        raise ValueError("global_hard_mask should be either True or False.")

assert args.global_hard_mask == eval_args.global_hard_mask, f"global_hard_mask:{args.global_hard_mask} in args and eval_args:{eval_args.global_hard_mask} are different."

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(args.seed)
random.seed(args.seed)
tick = time.time()

model_full_name = args.model_name.split("/")[-1].lower() #.split("-")[0]
model_names = ["llama-3", "llama-2", "llama", "opt", "vicuna"]      # WARNING: "llama-3" must be registed here. otherwise, f**king shit ppl will appears!!!
model_name = next((name for name in model_names if name in model_full_name), model_full_name)

model = get_llm(args.model_name, model_name, max_seq_length=eval_args.max_seq_length, dtype=eval_args.dtype)
tokenizer = get_tokenizer(args.model_name)

gpu_num = torch.cuda.device_count()
logger.info('Number of available gpu: {}'.format(gpu_num))

gpu_info = [f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in range(gpu_num)]
logger.info('GPUs: {}'.format(', '.join(gpu_info)))

device_map = model.hf_device_map
print(device_map)

data_device = torch.device('cuda:0') if "model.embed_tokens" not in device_map else device_map["model.embed_tokens"]


# dataset
logger.info('Loading test dataset {} for evaluation'.format(args.test_data_name))
test_data_module = get_data_module(args.test_data_name)(model_name, tokenizer=tokenizer, split='test',
                                                    # max_seq_length=args.max_seq_length,
                                                    max_seq_length=eval_args.max_seq_length,
                                                    )
test_data_module['batch_size'] = eval_args.batch_size
test_loader = torch.utils.data.DataLoader(**test_data_module)
torch.cuda.empty_cache()


model.eval()

# loading the prune info
prune_info_path = os.path.join(exp_dir, "prune_info.pt") if eval_args.pruned_checkpoint is None else os.path.join(exp_dir, f"ckpt_{eval_args.pruned_checkpoint}_prune_info.pt")
logger.info(f"Load prune_info from {prune_info_path}")
prune_info = torch.load(prune_info_path, map_location=torch.device('cpu'))
total_iters = math.floor(args.dataset_size / args.train_batch_size) * args.n_epoches
end_step = prune_info['end_step'] if 'end_step' in prune_info else total_iters
prune_rate = prune_info['prune_rate']

# construction from the prune info
layer_num = len(model.model.layers)
for i in range(layer_num):
    layer = model.model.layers[i]
    
    layer_device = next(iter(layer.parameters())).device
    wrapped_llama_layer = WrappedLlaMaLayer(layer, init_type="wanda-sp", K=args.K, device=layer_device)
    # get layer info
    layer_info = prune_info['layer_{}'.format(i)]
    if layer_info is None:
        logger.info(f"layer_{i}'s info is None. This layer is skipped.")
        continue
    wrapped_llama_layer.init_params((layer_info['attn_scores'], layer_info['mlp_scores']), (layer_info['init_attn_mask'], layer_info['init_mlp_mask']))
    model.model.layers[i] = wrapped_llama_layer

del layer_info
torch.cuda.empty_cache()
    
# construct pruner
pruner = ChannelPGPruner(model, tokenizer, total_iters, logger, args.train_data_name, 
                        prune_rate_start=args.prune_rate_start,
                        prune_rate_target=args.prune_rate_target,
                        prune_start_iter_percentage=args.prune_start_iter_percentage,
                        prune_end_iter_percentage=args.prune_end_iter_percentage,
                        criterion=nn.CrossEntropyLoss(),
                        K=args.K, mode='test',
                        # score_lr=args.score_lr,
                        attn_score_lr=args.attn_score_lr,
                        mlp_score_lr=args.mlp_score_lr,
                        ma_window_size=args.ma_window_size,
                        penalty_lamda_init=args.penalty_lamda_init,
                        penalty_lamda_final=args.penalty_lamda_final,
                        summary_writer=None,
                        DEBUG=args.debug,
                        global_hard_mask=eval_args.global_hard_mask,
                        )
torch.cuda.empty_cache()

logger.info("Check model sparsity of init_mask")
pruner.check_sparsity("init_mask")

logger.info("Check model sparsity of hard_mask")
pruner.check_sparsity("hard_mask")


pruner.set_mode('test')
logger.info(f"In seqlen = {pruner.subnet.seqlen}")
torch.cuda.empty_cache()

logger.info(f"global hard mask: {eval_args.global_hard_mask}")

# get the pruned model
pruned_model = pruner.structure_prune()
sparsity = check_llm_sparsity(pruned_model)
logger.info(f"Sparsity of the model after pruning: {sparsity}")
logger.info(f"model parameter {sum(p.numel() for p in pruned_model.parameters()) / 1000 ** 3:.2f}B")

del model, pruner
torch.cuda.empty_cache()

logger.info(f"prune model dtype is {pruned_model.dtype}")

ppl_pruned_model = llm_eval_ppl(pruned_model, test_loader, data_device)
logger.info(f"Perplexity of the pruned model: {ppl_pruned_model}")

task_list = eval_args.tasks
tasks = "+".join(task_list)
num_shot = eval_args.num_shot
is_accelerate = False
results = eval_zero_shot(model_name=args.model_name, model=pruned_model, tokenizer=tokenizer, task_list=task_list, num_fewshot=num_shot, use_accelerate=is_accelerate, batch_size=eval_args.batch_size)
results['perplexity'] = ppl_pruned_model
dumped = json.dumps(results, indent=2)

logger.info(dumped)

# write the results to a file
results_file_prefix = f"checkpoint-{eval_args.pruned_checkpoint}" if eval_args.pruned_checkpoint is not None else "final"
with open(os.path.join(eval_dir, f"{results_file_prefix}_pruned_model_{tasks}_{num_shot}shot_results.json"), 'w') as f:
    json.dump(results, f, indent=2)