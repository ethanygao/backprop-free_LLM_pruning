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

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from pgpruning import *
from modelutils import *
from datautils import get_data_module, map_tensors
from init import *
from evalutils import llm_eval_ppl

from logger import get_logger, date_uid

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

def pruning(pruner, train_loader, test_loader, device, logger, args, **kwargs):
    logger.info('Training ...')
            
    pruner.set_score_no_grad()
    
    step = 0
    for _ in range(args.n_epoches):
        for batch in tqdm(iter(train_loader), desc=f'Epoch {_ + 1}/{args.n_epoches}'):
            if not args.debug and step % args.eval_per_steps == 0:
                logger.info('\n')
                logger.info('Steps: {}/{}'.format(step, total_iters))
                pruner.log_score_state()

                ppl_hard = eval_ppl(pruner, test_loader, device, logger, args, mask_type='hard_mask')

                # save prune info
                if not args.not_save_checkpoint and step != 0:
                    logger.info(f"saving pruning info in checkpoint {step}...")
                    prune_info = pruner.get_prune_info()
                    torch.save(prune_info, os.path.join(exp_dir, f'ckpt_{step}_prune_info.pt'))

                torch.cuda.empty_cache()
            
            pruner.set_mode('train')
            
            """
            inp = torch.squeeze(batch['input_ids'], 1).to(device)
            target = torch.squeeze(batch['input_ids'][:, 1:], 1).to(device)
            causal_attention_mask = torch.squeeze(batch['attention_mask'], 1).to(device)
                        
            pruner.prune_pg(inp, target, causal_attention_mask)
            del inp, target, causal_attention_mask
            """
            batch = map_tensors(batch, device)
            pruner.prune_pg(batch)
            del batch
            
            torch.cuda.empty_cache()
            step += 1

    logger.info('\n')

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, help='Large language model to load; pass `meta-llama/Llama-2-7b-hf` for example.')
    
    parser.add_argument('--train-data-name', default="c4", type=str, 
                        choices=['wikitext2', 'c4'], 
                        help='Where to extract calibration data from.')
    parser.add_argument('--test-data-name', default="wikitext2", type=str, choices=['wikitext2', 'c4'], help='Where to evaluate the sparse model.')

    parser.add_argument('--seed', type=int, default=42, help='Seed for sampling the calibration data.')
    parser.add_argument("--dataset-size", default=None, type=int, help="dataset size (default: 50000)")
    parser.add_argument("--max-seq-length", default=2048, type=int, help="dataset size (default: 2048)")
    parser.add_argument("--init-seqlen", default=1024, type=int, help="dataset size (default: 2048)")
    parser.add_argument('--n-workers', type=int, default=8, help='Number of workers to load data.')
    parser.add_argument("--train-batch-size", default=8, type=int, help="batch size (default: 256)")
    parser.add_argument("--test-batch-size", default=16, type=int, help="batch size (default: 256)")
    parser.add_argument("--test-seqlen", default=128, type=int, help="ppl test seqlen")

    parser.add_argument("--warmup-length", default=0, type=int, help="Number of warmup iterations")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay (default: 1e-4)")

    parser.add_argument("--attn-score-lr", default=1e-3, type=float, help="initial atention head score learning rate")
    parser.add_argument("--mlp-score-lr", default=1e-3, type=float, help="initial mlp head score learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for sgd")
    parser.add_argument("--lr-policy", default="cosine", choices=["cosine", "poly", "step", "exp", "const"], help="Policy for the learning rate")
    parser.add_argument("--lr-adjust", default=30, type=int, help="Interval to drop lr")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="Multistep multiplier")

    parser.add_argument("--prune-rate-target", default=0.5, type=float, help="Amount of pruning to do during sparse training")
    parser.add_argument("--prune-rate-start", default=1.0, type=float, help="Amount of pruning rate for start")
    parser.add_argument("--prune-start-iter-percentage", type=float, default=0, help="start of pruning")    # not used actually
    parser.add_argument("--prune-end-iter-percentage", type=float, default=0.6, help="end of pruning")      # not used actually
    parser.add_argument("--score-init-constant", type=float, default=None, help='initial value of the masks')
    parser.add_argument("--init-type", type=str, default="wanda-sp", choices=['wanda-sp', 'llm-pruner', 'llm-pruner-skip', 'random'], help="init method, wanda default")
    parser.add_argument("--llm-pruner-dir", type=str, help="LLM-Pruner pruning info directory")
    parser.add_argument("--init-rate", type=float, default=0.0, help="init method's prune rate")
    parser.add_argument('--init-data-name', default="c4", type=str, 
                        choices=['wikitext2', 'c4'], 
                        help='Where to extract init calibration data from.')
    parser.add_argument("--score-from-metric", type=str, default="const", choices=['const', 'norm', 'sigmap', 'hybrid'], help="init scores from wanda/aw metric, norm default")
    parser.add_argument('--global-hard-mask', type=str, default=None, choices=['True', 'False'] ,help='Whether use global hard mask')

    # init from previous prune info checkpoint
    parser.add_argument('--init-prune-info', type=str, default=None, help='Path to previous saved prune info.')
    
    parser.add_argument("--K", type=int, default=2, help="Sample K net replications")
    parser.add_argument("--nsamples", type=int, default=128, help="number of samples for wanda init forward pass")

    parser.add_argument("--eval-per-steps", type=int, default=100, help="do eval on x steps when training")

    # typically we do not need this as we can randomly generate arbitrary number of sequences using --dataset-size
    parser.add_argument("--n-epoches", type=int, default=1, help="how many epoches to train") 

    # moving average window size
    parser.add_argument("--ma-window-size", type=int, default=10, help="moving average window size")
    # penalty lamda init/final, not used actually
    parser.add_argument("--penalty-lamda-init", type=float, default=0.01, help="penalty lamda init. if is set as a negative, then penalty lamda will be 0 for all iterations.")
    parser.add_argument("--penalty-lamda-final", type=float, default=0.01, help="penalty lamda final.")
    
    parser.add_argument('--save-folder', type=str, default='exp', help='Path to saved model.')
    parser.add_argument('--exp-name', type=str, default='test', help='Path to saved model.')

    parser.add_argument('--not-save-model', action="store_true", help='Save model.')
    parser.add_argument('--not-save-checkpoint', action="store_true", help='Save checkpoint.')
    parser.add_argument('--debug', action="store_true", help='Debug mode.')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp32', 'fp16','bf16'], help='Data type.')
    
    parser.add_argument('--cancel-ma-baseline', action="store_true", help='Cancel the moving average baseline.')

    args = parser.parse_args()

    # prepare logging
    save_folder = args.save_folder
    os.makedirs(args.save_folder) if not os.path.exists(args.save_folder) else None
    exp_dir = os.path.join(args.save_folder, args.exp_name, date_uid())
    os.makedirs(exp_dir) if not os.path.exists(exp_dir) else None

    # tensorboard logging
    writer = SummaryWriter(log_dir=exp_dir)
    
    # save args to a conf file in the exp_dir
    ## dataset_size might be modified, donot save it soo early
    # with open(os.path.join(exp_dir, 'params.json'), 'w') as f:
    #     json.dump(vars(args), f, indent=2)
    logger = get_logger(exp_dir)
    logger.info(args)

    tick = time.time()
    if os.path.exists(args.model_name):
        print("Path exists")
    else:
        print("Path does not exist")

    model_full_name = args.model_name.split("/")[-1].lower() #.split("-")[0]
    model_names = ["llama-3", "llama-2", "llama", "vicuna"]
    model_name = next((name for name in model_names if name in model_full_name), model_full_name)

    model = get_llm(args.model_name, model_name, max_seq_length=args.max_seq_length, dtype=args.dtype)
    tokenizer = get_tokenizer(args.model_name)
    logger.info('load model and tokenizer in {:.3f}s'.format(time.time() - tick))  

    before_pruning_parameters = sum(p.numel() for p in model.parameters())

    gpu_num = torch.cuda.device_count()
    logger.info('Number of available gpu: {}'.format(gpu_num))

    gpu_info = [f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in range(gpu_num)]
    logger.info('GPUs: {}'.format(', '.join(gpu_info)))

    device_map = model.hf_device_map
    print(device_map)

    data_device = torch.device('cuda:0') if "model.embed_tokens" not in device_map else device_map["model.embed_tokens"]
    
    # dataset
    logger.info('Loading train dataset {} for calibration'.format(args.train_data_name))
    train_data_module = get_data_module(args.train_data_name)(model_name, tokenizer=tokenizer, split='train',
                                                        max_seq_length=args.max_seq_length,
                                                        max_samples=args.dataset_size,
                                                        seed=args.seed
                                                        )
    train_data_module['batch_size'] = args.train_batch_size
    train_loader = torch.utils.data.DataLoader(**train_data_module)
                                               
    logger.info('Loading test dataset {} for evaluation'.format(args.test_data_name))
    test_data_module = get_data_module(args.test_data_name)(model_name, tokenizer=tokenizer, split='test',
                                                        max_seq_length=args.test_seqlen,
                                                        )
    test_data_module['batch_size'] = args.test_batch_size
    test_loader = torch.utils.data.DataLoader(**test_data_module)
    
    torch.cuda.empty_cache()
    
    model.eval()

    if args.init_rate > 0:
        logger.info('Loading init dataset {} for evaluation'.format(args.init_data_name))
        model_seqlen = model.seqlen
        model.seqlen = args.init_seqlen
        logger.info(f"Model seqlen for init dataset: {model.seqlen}")
        init_data_module = get_data_module(args.init_data_name)(model_name, tokenizer=tokenizer, split='train',
                                                        max_seq_length=model.seqlen,
                                                        max_samples=args.nsamples,
                                                        seed=args.seed
                                                        )
        # train_dataset, eval_dataset = init_data_module["train_dataset"], init_data_module["eval_dataset"]
        # del init_data_module["train_dataset"], init_data_module["eval_dataset"]
        # init_data_module['batch_size'] = 1
        # init_loader = torch.utils.data.DataLoader(train_dataset, **init_data_module)
        init_data_module['batch_size'] = 1
        init_loader = torch.utils.data.DataLoader(**init_data_module)
        
        # # set random seed
        # init_dataset.random.seed(args.seed)

        # different initialization strategy
        if args.init_type == "wanda-sp":
            t = time.time()
            wanda_sp_init(args, model, init_loader, device=data_device)
            logger.info(f"Init model by Wanda-sp in {time.time() - t:.3f}s")
        elif args.init_type == "llm-pruner":
            llm_pruner_init(args, model)
        elif args.init_type == "llm-pruner-skip":
            llm_pruner_init_skip(args, model)
        elif args.init_type == "random":
            random_init(args, model)
        else:
            raise ValueError("Unsupported init method: {}".format(args.init_type))
        
        model.seqlen = model_seqlen
        del init_loader
        torch.cuda.empty_cache()
    elif args.init_prune_info is not None:
        logger.info(f"Init from prune info: {args.init_prune_info}")
        prune_info_init(args, model, args.init_prune_info)
    else:
        raise ValueError("init rate should be larger than 0 or init prune info is provided")
    
    torch.cuda.empty_cache()

    # total_iters = math.floor(args.dataset_size / args.batch_size) * args.n_epoches
    total_iters = math.floor(len(train_loader.dataset) / args.train_batch_size) * args.n_epoches
    if args.dataset_size is None:
        args.dataset_size = len(train_loader.dataset)

    # save args to a conf file in the exp_dir
    with open(os.path.join(exp_dir, 'params.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)    

    # wanda-sp init: global hard mask; llm-pruner init: local hard mask
    # `llm-pruner` use local prune, but global hard mask is better when it is used for initialization here.
    logger.info(f"args.global_hard_mask = {args.global_hard_mask}")
    if args.global_hard_mask is None:
        global_hard_mask = True if args.init_type in ["wanda-sp", "random"] else False
    else:
        if args.global_hard_mask == "True":
            global_hard_mask = True
        elif args.global_hard_mask == "False":
            global_hard_mask = False
        else:
            raise ValueError("Unsupported global hard mask: {}".format(args.global_hard_mask))
    logger.info("Global hard mask: {}".format(global_hard_mask))
    
    pruner = ChannelPGPruner(model, tokenizer, total_iters, logger, args.train_data_name, 
                            prune_rate_start=args.prune_rate_start,
                            prune_rate_target=args.prune_rate_target,
                            prune_start_iter_percentage=args.prune_start_iter_percentage,   # not used actually
                            prune_end_iter_percentage=args.prune_end_iter_percentage,   # not used actually
                            criterion=nn.CrossEntropyLoss(),
                            K=args.K, mode='train',
                            lr_policy=args.lr_policy,
                            attn_score_lr=args.attn_score_lr,
                            mlp_score_lr=args.mlp_score_lr,
                            ma_window_size=args.ma_window_size,
                            penalty_lamda_init=args.penalty_lamda_init,
                            penalty_lamda_final=args.penalty_lamda_final,
                            summary_writer=writer,
                            DEBUG=args.debug,
                            global_hard_mask=global_hard_mask,
                            cancel_ma_baseline=args.cancel_ma_baseline,
                            )
    
    real_init_rate = pruner.check_sparsity("init_mask")
    # assert abs(real_init_rate - args.init_rate) < 1e-2, "real init rate is not equal to the expected init rate"

    if not args.debug:        
        logger.info('Evaluating initial mask before training...')
        ppl_init_mask = eval_ppl(pruner, test_loader, data_device, logger, args, mask_type="init_mask")

    torch.cuda.empty_cache()
    
    # policy gradient pruning begins!
    pruning(pruner, train_loader, test_loader, data_device, logger, args, exp_dir=exp_dir)

    logger.info('prune network in {:.3f}s'.format(time.time() - tick))

    pruner.log_score_state()

    logger.info("saving pruning info...")
    prune_info = pruner.get_prune_info()
    torch.save(prune_info, os.path.join(exp_dir, 'prune_info.pt'))

    # pruner.set_mode('test')
    # ppl_soft = eval_ppl(pruner, test_loader, data_device, logger, args, mask_type="soft_mask")
    ppl_hard = eval_ppl(pruner, test_loader, data_device, logger, args, mask_type="hard_mask")
    
    logger.info("Compress Model...")
    pruned_model = pruner.structure_prune()
    sparsity = check_llm_sparsity(pruned_model)
    logger.info("Prune rate means the percentage of the saved parameters in the model")
    logger.info(f"Prune rate of the model after pruning: {sparsity}")
    after_pruning_parameters = sum(p.numel() for p in pruned_model.parameters())
    logger.info(f"model parameter {sum(p.numel() for p in pruned_model.parameters()) / 1000 ** 3:.2f}B")
    logger.info("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))

    ppl_pruned_model = llm_eval_ppl(pruned_model, test_loader, data_device)
    logger.info(f"Perplexity of the pruned model: {ppl_pruned_model}")

    if not args.not_save_model:
        # print("saving pruning info...")
        # prune_info = pruner.get_prune_info()
        # torch.save(prune_info, os.path.join(exp_dir, 'prune_info.pt'))
        # pass
        logger.info('Saving pruned model...')
        saved_model_path = os.path.join(exp_dir, "llm_weight")
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)
        pruned_model.save_pretrained(saved_model_path)
        tokenizer.save_pretrained(saved_model_path)