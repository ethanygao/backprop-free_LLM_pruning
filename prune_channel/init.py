import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import numpy as np
import math
import random
import os
import json

import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *
from modelutils import *


# layerwrapper
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        
    def free(self):
        self.scaler_row = None
        torch.cuda.empty_cache()

def prepare_calibration_input(model, dataloader, device, nsamples=128):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    # TODO: check whether nsamples is larger than the number of batches in dataloader

    ## *** i dont understand why wanda will do like this***
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError    # to make the model forward stop at the first layer
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        # print(batch[0].shape)
        if cache['i'] >= nsamples:
            break
        try:
            # model(batch[0].squeeze(1).to(device))
            model(batch["input_ids"].squeeze(1).to(device))
        except ValueError:
            pass 
        # model(batch[0].to(device))  # too many values to unpack (expected 2)

    
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids

def wanda_sp_init(args, model, dataloader, device=torch.device("cuda:0")):
    """
    This is the init method based on input-output cosine distance metric 
    """
    assert args.init_rate > 0, "init_rate should be larger than 0"
    assert args.score_init_constant > 0, "score_init_constant should be larger than 0"

    use_cache = model.config.use_cache
    model.config.use_cache = False

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        num_key_value_heads = layer.self_attn.num_key_value_heads
        num_key_value_groups = layer.self_attn.num_key_value_groups

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        
        attn_metric = None
        mlp_metric = None
        attn_init_mask = None
        mlp_init_mask = None
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            
            if name == 'self_attn.o_proj':
                W_metric = W_metric.mean(axis=0).reshape(-1, 128).sum(dim=1)    # importance score of each head
                # W_metric = W_metric.reshape(num_key_value_heads, -1).sum(dim=1) # importance score of each key-value head
                W_metric = W_metric.reshape(num_key_value_heads, -1).mean(dim=1) # importance score of each key-value head
                thresh = torch.sort(W_metric.cuda())[0][int((1 - args.init_rate)*num_key_value_heads)].cpu()
                W_mask = (W_metric>=thresh)
                # save the metric & mask
                attn_metric = W_metric
                attn_init_mask = W_mask
            else:
                W_metric = W_metric.mean(axis=0)
                thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*(1 - args.init_rate))].cpu()
                W_mask = (W_metric>=thresh)
                # save the metric & mask
                mlp_metric = W_metric
                mlp_init_mask = W_mask
        
            wrapped_layers[name].free()
        
        init_attn_scores = torch.zeros_like(attn_metric)
        init_mlp_scores = torch.zeros_like(mlp_metric)
        if args.score_from_metric == 'const':
            init_attn_scores[attn_init_mask] = args.score_init_constant
            init_attn_scores[~attn_init_mask] = 1-args.score_init_constant
            init_mlp_scores[mlp_init_mask] = args.score_init_constant
            init_mlp_scores[~mlp_init_mask] = 1-args.score_init_constant
        elif args.score_from_metric == 'norm':
            init_attn_scores = normalize_data(attn_metric, args.init_rate).clamp_(0,1)
            init_mlp_scores = normalize_data(mlp_metric, args.init_rate).clamp_(0,1)
        elif args.score_from_metric == 'sigmap':
            init_attn_scores = sigmap(attn_metric, args.init_rate).clamp_(0,1)
            init_mlp_scores = sigmap(mlp_metric, args.init_rate).clamp_(0,1)
        elif args.score_from_metric == 'hybrid':
            init_attn_scores = sigmap(attn_metric, args.init_rate).clamp_(0,1)
            init_mlp_scores = normalize_data(mlp_metric, args.init_rate).clamp_(0,1)
        else:
            raise NotImplementedError(f"score_from_metric {args.score_from_metric} is not implemented")
        
        # replace layer with the wrapped layer after init
        layer_device = next(iter(layer.parameters())).device
        wrapped_llama_layer = WrappedLlaMaLayer(layer, init_type="wanda-sp", K=args.K, device=layer_device)
        wrapped_llama_layer.init_params((init_attn_scores, init_mlp_scores), (attn_init_mask, mlp_init_mask))
        model.model.layers[i] = wrapped_llama_layer

        ## the pruned output as input to the next layer, as vanilla wanda ##
        wrapped_llama_layer.set_mask("init_mask")
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = wrapped_llama_layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps 
            
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

def llm_pruner_init(args, model):
    """
    This is the init method for LLM pruner.
    This init way will give a constant score to the llm-pruner' skipped layers,
    which mean it take these skipped layers into consideration.
    """
    llm_pruner_dir = args.llm_pruner_dir
    assert os.path.exists(llm_pruner_dir), f"llm_pruner_dir {llm_pruner_dir} does not exist"
    # read config file of llm-pruner 
    llm_pruner_config = os.path.join(llm_pruner_dir, "description.json")
    with open(llm_pruner_config, "r") as f:
        llm_pruner_config = json.load(f)

    # check configurations
    llm_pruner_base_model = llm_pruner_config["base_model"].split("/")[-1]
    args_model_name = args.model_name.split("/")[-1]
    assert llm_pruner_base_model == args_model_name, "base_model in llm_pruner_config is not consistent with args.model_name"
    # assert llm_pruner_config["base_model"] == args.model_name, "base_model in llm_pruner_config is not consistent with args.model_name"
    assert llm_pruner_config["cal_dataset"] == args.init_data_name, "cal_dataset in llm_pruner_config is not consistent with args.init_data_name"
    # assert abs(llm_pruner_config["final_prune_rate"] - args.init_rate) < 0.03, "llm-pruner's final prune is not consistent with args.init_rate"

    llm_pruner_info_path = os.path.join(llm_pruner_dir, "prune_info.pth")
    assert os.path.exists(llm_pruner_info_path), f"llm_pruner_info path {llm_pruner_info_path} does not exist"
    
    llm_pruner_info = torch.load(llm_pruner_info_path)
    llm_pruner_info_keys = llm_pruner_info.keys()

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        self_attn_num_heads = layer.self_attn.num_key_value_heads
        mlp_intermediate_size = layer.mlp.intermediate_size

        has_attn_mask = f"layer.{i}.self_attn.mask" in llm_pruner_info_keys
        has_mlp_mask = f"layer.{i}.mlp.mask" in llm_pruner_info_keys

        if has_attn_mask and has_mlp_mask:
            attn_mask = llm_pruner_info[f"layer.{i}.self_attn.mask"]
            mlp_mask = llm_pruner_info[f"layer.{i}.mlp.mask"]
            attn_metric = llm_pruner_info[f"layer.{i}.self_attn.importance"]
            mlp_metric = llm_pruner_info[f"layer.{i}.mlp.importance"]
            
            assert attn_mask.shape == (self_attn_num_heads,), f"attn_mask shape {attn_mask.shape} is not correct"
            assert mlp_mask.shape == (mlp_intermediate_size,), f"mlp_mask shape {mlp_mask.shape} is not correct"

        elif not has_attn_mask and not has_mlp_mask:
            # this means llm-pruner skipped this layer
            attn_mask = torch.ones((self_attn_num_heads), dtype=torch.bool)
            mlp_mask = torch.ones((mlp_intermediate_size), dtype=torch.bool)
            # attn_metric = None
            # mlp_metric = None
            attn_metric = torch.ones((self_attn_num_heads), dtype=torch.float)
            mlp_metric = torch.ones((mlp_intermediate_size), dtype=torch.float)

        else:
            raise ValueError(f"layer {i} should have or not have self_attn_mask and mlp_mask at the same time!")
        
        print(f"LLM pruner init layer {i} prune rate: self_attn {attn_mask.float().mean():.4f}, mlp {mlp_mask.float().mean():.4f}")
        
        init_attn_scores = torch.ones_like(attn_mask).float()
        init_mlp_scores = torch.ones_like(mlp_mask).float()
        if args.score_from_metric == 'const':
            init_attn_scores[attn_mask] = args.score_init_constant
            init_attn_scores[~attn_mask] = 1-args.score_init_constant
            init_mlp_scores[mlp_mask] = args.score_init_constant
            init_mlp_scores[~mlp_mask] = 1-args.score_init_constant
        elif args.score_from_metric == 'sigmap':
            init_attn_scores = sigmap(attn_metric, args.init_rate).clamp(0,1) if attn_metric is not None else init_attn_scores
            init_mlp_scores = sigmap(mlp_metric, args.init_rate).clamp(0,1) if mlp_metric is not None else init_mlp_scores
        else:
            raise NotImplementedError(f"score_from_metric {args.score_from_metric} is not implemented")
        

        # replace layer with the wrapped layer after init
        layer_device = next(iter(layer.parameters())).device
        wrapped_llama_layer = WrappedLlaMaLayer(layer, init_type="wanda-sp", K=args.K, device=layer_device)
        wrapped_llama_layer.init_params((init_attn_scores, init_mlp_scores), (attn_mask, mlp_mask))
        model.model.layers[i] = wrapped_llama_layer

def llm_pruner_init_skip(args, model):
    """
    This is the init method for LLM pruner.
    This init way will ignore the llm-pruner' skipped layers, without wrapping them
    """
    llm_pruner_dir = args.llm_pruner_dir
    assert os.path.exists(llm_pruner_dir), f"llm_pruner_dir {llm_pruner_dir} does not exist"
    # read config file of llm-pruner 
    llm_pruner_config = os.path.join(llm_pruner_dir, "description.json")
    with open(llm_pruner_config, "r") as f:
        llm_pruner_config = json.load(f)

    # check configurations
    llm_pruner_base_model = llm_pruner_config["base_model"].split("/")[-1]
    args_model_name = args.model_name.split("/")[-1]
    assert llm_pruner_base_model == args_model_name, "base_model in llm_pruner_config is not consistent with args.model_name"
    assert llm_pruner_config["cal_dataset"] == args.init_data_name, "cal_dataset in llm_pruner_config is not consistent with args.init_data_name"
    # assert abs(1-llm_pruner_config["pruning_ratio"] - args.init_rate) < 0.001, "llm-pruner's pruning rate is not consistent with args.init_rate"

    llm_pruner_info_path = os.path.join(llm_pruner_dir, "prune_info.pth")
    assert os.path.exists(llm_pruner_info_path), f"llm_pruner_info path {llm_pruner_info_path} does not exist"
    
    llm_pruner_info = torch.load(llm_pruner_info_path)
    llm_pruner_info_keys = llm_pruner_info.keys()

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        self_attn_num_heads = layer.self_attn.num_key_value_heads
        mlp_intermediate_size = layer.mlp.intermediate_size

        has_attn_mask = f"layer.{i}.self_attn.mask" in llm_pruner_info_keys
        has_mlp_mask = f"layer.{i}.mlp.mask" in llm_pruner_info_keys

        if has_attn_mask and has_mlp_mask:
            attn_mask = llm_pruner_info[f"layer.{i}.self_attn.mask"]
            mlp_mask = llm_pruner_info[f"layer.{i}.mlp.mask"]
            attn_metric = llm_pruner_info[f"layer.{i}.self_attn.importance"]
            mlp_metric = llm_pruner_info[f"layer.{i}.mlp.importance"]
            
            assert attn_mask.shape == (self_attn_num_heads,), f"attn_mask shape {attn_mask.shape} is not correct"
            assert mlp_mask.shape == (mlp_intermediate_size,), f"mlp_mask shape {mlp_mask.shape} is not correct"

        elif not has_attn_mask and not has_mlp_mask:
            print(f"llm-pruner skipped layer {i}")
            continue
        else:
            raise ValueError(f"layer {i} should have or not have self_attn_mask and mlp_mask at the same time!")
        
        print(f"LLM pruner init layer {i} prune rate: self_attn {attn_mask.float().mean():.4f}, mlp {mlp_mask.float().mean():.4f}")
        
        init_attn_scores = torch.ones_like(attn_mask).float()
        init_mlp_scores = torch.ones_like(mlp_mask).float()
        if args.score_from_metric == 'const':
            init_attn_scores[attn_mask] = args.score_init_constant
            init_attn_scores[~attn_mask] = 1-args.score_init_constant
            init_mlp_scores[mlp_mask] = args.score_init_constant
            init_mlp_scores[~mlp_mask] = 1-args.score_init_constant
        elif args.score_from_metric == 'sigmap':
            init_attn_scores = sigmap(attn_metric, args.init_rate).clamp(0,1) if attn_metric is not None else init_attn_scores
            init_mlp_scores = sigmap(mlp_metric, args.init_rate).clamp(0,1) if mlp_metric is not None else init_mlp_scores
        elif args.score_from_metric == 'norm':
            init_attn_scores = normalize_data(attn_metric, args.init_rate).clamp_(0,1) if attn_metric is not None else init_attn_scores
            init_mlp_scores = normalize_data(mlp_metric, args.init_rate).clamp_(0,1) if mlp_metric is not None else init_mlp_scores
        else:
            raise NotImplementedError(f"score_from_metric {args.score_from_metric} is not implemented")
        

        # replace layer with the wrapped layer after init
        layer_device = next(iter(layer.parameters())).device
        wrapped_llama_layer = WrappedLlaMaLayer(layer, init_type="wanda-sp", K=args.K, device=layer_device)
        wrapped_llama_layer.init_params((init_attn_scores, init_mlp_scores), (attn_mask, mlp_mask))
        model.model.layers[i] = wrapped_llama_layer

def random_init(args, model):
    """
    This is the random init method.
    """
    assert args.init_rate > 0, "init_rate should be larger than 0"

    use_cache = model.config.use_cache
    model.config.use_cache = False

    def random_score(size, rate):
        rn = torch.randn(size)
        return sigmap(rn, rate).clamp_(0,1)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        self_attn_num_heads = layer.self_attn.num_heads
        num_key_value_heads = layer.self_attn.num_key_value_heads
        mlp_intermediate_size = layer.mlp.intermediate_size

        # random attn score and mask
        init_attn_scores = random_score((num_key_value_heads,), args.init_rate)
        attn_thresh = torch.sort(init_attn_scores)[0][int(num_key_value_heads*(1-args.init_rate))]
        attn_init_mask = (init_attn_scores >= attn_thresh)

        # random mlp score and mask
        init_mlp_scores = random_score((mlp_intermediate_size,), args.init_rate)
        mlp_thresh = torch.sort(init_mlp_scores)[0][int(mlp_intermediate_size*(1-args.init_rate))]
        mlp_init_mask = (init_mlp_scores >= mlp_thresh)

        # replace layer with the wrapped layer after init
        layer_device = next(iter(layer.parameters())).device
        wrapped_llama_layer = WrappedLlaMaLayer(layer, init_type="random", K=args.K, device=layer_device)
        # the init mask is layerwise mask, not used in the random init method
        wrapped_llama_layer.init_params((init_attn_scores, init_mlp_scores), (attn_init_mask, mlp_init_mask))
        model.model.layers[i] = wrapped_llama_layer

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def self_init(model, init_rate, global_hard_mask):
    """
    This is the self init method
    """
    assert init_rate > 0, "init_rate should be larger than 0"

    use_cache = model.config.use_cache
    model.config.use_cache = False

    def get_global_hard_mask_thresh(model, rate):
        all_attn_scores = []
        all_mlp_scores = []
        for layer in model.model.layers:
            if isinstance(layer, WrappedLlaMaLayer):
                all_attn_scores.append(layer.attn_scores.flatten().cpu())
                all_mlp_scores.append(layer.mlp_scores.flatten().cpu())

        combined_attn_scores = torch.cat(all_attn_scores)
        combined_mlp_scores = torch.cat(all_mlp_scores)

        attn_thresh = torch.sort(combined_attn_scores.cuda())[0][int(combined_attn_scores.numel() * (1 - rate))].cpu()
        mlp_thresh = torch.sort(combined_mlp_scores.cuda())[0][int(combined_mlp_scores.numel() * (1 - rate))].cpu()

        attn_thresh, mlp_thresh = attn_thresh.item(), mlp_thresh.item()
        return attn_thresh, mlp_thresh

    # get hard thresh in 'init_rate' from old score
    old_attn_thresh, old_mlp_thresh = get_global_hard_mask_thresh(model, init_rate)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        if isinstance(layer, WrappedLlaMaLayer):
            old_attn_scores = layer.attn_scores
            old_mlp_scores = layer.mlp_scores
            if global_hard_mask:
                attn_thresh, mlp_thresh = old_attn_thresh, old_mlp_thresh
            else:
                attn_thresh = torch.sort(old_attn_scores.cuda())[0][int(old_attn_scores.numel() * (1 - init_rate))].cpu()
                mlp_thresh = torch.sort(old_mlp_scores.cuda())[0][int(old_mlp_scores.numel() * (1 - init_rate))].cpu()

            # get init mask from old score
            attn_init_mask = (old_attn_scores >= attn_thresh)
            mlp_init_mask = (old_mlp_scores >= mlp_thresh)

            init_attn_scores = sigmap(old_attn_scores, init_rate).clamp_(0,1)
            init_mlp_scores = sigmap(old_mlp_scores, init_rate).clamp_(0,1)
            # the init mask might not be the same as the hard mask from new score
            layer.init_params((init_attn_scores, init_mlp_scores), (attn_init_mask, mlp_init_mask))

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    
def prune_info_init(args, model, prune_info_path):
    """
    This is the init method for prune_info.pth
    """
    assert os.path.exists(prune_info_path), f"prune_info_path {prune_info_path} does not exist"
    prune_info = torch.load(prune_info_path)
    # prune_info_keys = prune_info.keys()
    prune_rate = prune_info['prune_rate']
    assert args.prune_rate_start == prune_rate, f"prune_rate_start {args.prune_rate_start} is not consistent with prune_rate {prune_rate}"

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        self_attn_num_heads = layer.self_attn.num_key_value_heads
        mlp_intermediate_size = layer.mlp.intermediate_size

        layer_device = next(iter(layer.parameters())).device
        wrapped_llama_layer = WrappedLlaMaLayer(layer, init_type="prune-info", K=args.K, device=layer_device)
        # get layer info
        layer_info = prune_info['layer_{}'.format(i)]
        if layer_info is None:
            print(f"layer_{i}'s info is None. This layer is skipped.")
            continue
        wrapped_llama_layer.init_params((layer_info['attn_scores'], layer_info['mlp_scores']), (layer_info['hard_attn_mask'], layer_info['hard_mlp_mask']))
        model.model.layers[i] = wrapped_llama_layer
    
    del prune_info
    torch.cuda.empty_cache()
