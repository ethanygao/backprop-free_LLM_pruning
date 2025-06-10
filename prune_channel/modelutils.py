import torch
import torch.nn as nn
import transformers
import numpy as np
import math
import sys
from copy import deepcopy
from typing import List, Optional, Tuple, Union

# from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, MistralForCausalLM
# new import
from models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, MistralForCausalLM

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_tokenizer(model_name):
    if "llama" in model_name.lower():
        if "llama-3" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
            # fix for transformer 4.28.0.dev0 compatibility
            if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
                try:
                    tokenizer.bos_token_id = 1
                    tokenizer.eos_token_id = 2
                except AttributeError:
                    pass
        # add pad token
        # if tokenizer.pad_token_id is None:
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    return tokenizer


DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
def get_llm(model_path, model_name, max_seq_length=None, cache_dir="llm_weights", dtype="bf16"):
    if model_name in ["llama-3", "llama-2", "llama", "vicuna"]:
        model = LlamaForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=DTYPE_MAP[dtype],     #torch.float16,
            cache_dir=cache_dir, 
            # low_cpu_mem_usage=True, 
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=DTYPE_MAP[dtype],     #torch.float16,
            cache_dir=cache_dir, 
            low_cpu_mem_usage=True, 
            device_map="auto"
        )
    
    if max_seq_length is None:
        model.seqlen = model.config.max_position_embeddings
    else:
        model.seqlen = max_seq_length

    return model
    
class WrappedLlaMaLayer(nn.Module):
    """
    wrapped llama layers for pg pruning
    """
    def __init__(self, layer, init_type="wanda-sp", K=2, device=torch.device("cuda:0")):
        super(WrappedLlaMaLayer, self).__init__()

        # assert isinstance(model, LlamaDecoderLayer), "model should be an instance of LlamaForCausalLM"

        self.init_type = init_type
        self.device = device

        # self.const_layer = layer
        self.layer = layer
        self.saved_self_attn_o_proj = layer.self_attn.o_proj.weight.data.clone()
        self.saved_mlp_down_proj = layer.mlp.down_proj.weight.data.clone()

        self.attn_scores = nn.Parameter(torch.Tensor(layer.self_attn.num_key_value_heads).to(device))
        self.mlp_scores = nn.Parameter(torch.Tensor(layer.mlp.intermediate_size).to(device))

        self.stored_attn_mask = torch.zeros((K, *self.attn_scores.shape), dtype=torch.bool).to(device)
        self.stored_mlp_mask = torch.zeros((K, *self.mlp_scores.shape), dtype=torch.bool).to(device)
        self.init_attn_mask = (torch.ones_like(self.attn_scores) == 1).cpu()
        self.init_mlp_mask = (torch.ones_like(self.mlp_scores) == 1).cpu()

        self.K = K
        self.index = -1

        # flag for whether inited
        self.inited = False
        
        # flags and function for different types of masks
        ## used for sample mask ## 
        def sample_mask():
            if not self.is_sampled:
                self.sampled_attn_mask = (torch.rand_like(self.attn_scores) < self.attn_scores)
                self.sampled_mlp_mask = (torch.rand_like(self.mlp_scores) < self.mlp_scores)
                self.is_sampled = True
            return self.sampled_attn_mask, self.sampled_mlp_mask
        self.sampled_attn_mask = None
        self.sampled_mlp_mask = None
        self.is_sampled = False
        
        self.functions = {
            "none_mask": lambda: (torch.ones_like(self.attn_scores), 
                                  torch.ones_like(self.mlp_scores)), 
            "soft_mask": lambda: (torch.rand_like(self.attn_scores) < self.attn_scores, 
                                  torch.rand_like(self.mlp_scores) < self.mlp_scores,),
            "hard_mask": lambda: (torch.abs(self.attn_scores) >= self.attn_thresh,
                                  torch.abs(self.mlp_scores) >= self.mlp_thresh),
            "init_mask": lambda: (self.init_attn_mask, self.init_mlp_mask),
            ## Soft mask but sample then remember it and use it for the rest of the process. ##
            ## Only for testing!!! ##
            "sampled_mask": sample_mask
        }
        
        
        self.mask_type = "none_mask"
        self.mask_func = self.functions[self.mask_type]
        self.attn_thresh = -1
        self.mlp_thresh = -1

    def init_params(self, init_scores, init_mask):
        # print("init Wrapped LlaMa params")
        self.attn_scores.data = init_scores[0].to(self.attn_scores.device)
        self.mlp_scores.data = init_scores[1].to(self.mlp_scores.device)
        self.init_attn_mask = init_mask[0].to(self.stored_attn_mask.device)
        self.init_mlp_mask = init_mask[1].to(self.stored_mlp_mask.device)
        self.inited = True

    def set_mask(self, mask_type, attn_thresh=-1, mlp_thresh=-1):
        assert mask_type in self.functions.keys(), f"mask type should be in {self.functions.keys()}"

        if mask_type == 'hard_mask':
            self.attn_thresh = attn_thresh
            self.mlp_thresh = mlp_thresh
        self.mask_type = mask_type
        self.mask_func = self.functions[mask_type]
    
    def get_mask(self, mask_type, attn_thresh=-1, mlp_thresh=-1):
        self.set_mask(mask_type, attn_thresh, mlp_thresh)
        return self.mask_func()

    def set_index(self, index):
        self.index = index

    def clear(self):
        self.attn_thresh = -1
        self.mlp_thresh = -1
        self.mask_type = "none_mask"
        self.mask_func = self.functions[self.mask_type]
        ## clear sampled mask ##
        self.sampled_attn_mask = None
        self.sampled_mlp_mask = None
        self.is_sampled = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # The input params list is copy from huggingface/transformer
        if not self.inited:
            raise ValueError("Scores and mask should be inited before forward pass")

        attn_mask, mlp_mask = self.mask_func()
        if self.training:
            self.stored_attn_mask[self.index] = attn_mask
            self.stored_mlp_mask[self.index] = mlp_mask
        
        # made_layer = self.makeup_layer(attn_mask, mlp_mask)
        # self.makeup_layer(attn_mask, mlp_mask)
        
        output = self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            ##### new added #####
            attn_head_mask=attn_mask,
            mlp_mask=mlp_mask,
            #####################
            **kwargs,
        )
        
        # self.recover_layer()
        return output


def check_llm_sparsity(model):
    """
    Check the sparsity of the weights in different layers of the model.
    
    Args:
        model (nn.Module): The model to check.
        
    Returns:
        float: Ratio of the count of non-zero weights to total parameters in the model.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    # TODO: Not suitable for llama3 GQA
    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            sub_count += W.numel()
            count += W.numel()
            if 'self_attn' in name:
                if 'o_proj' in name or 'q_proj' in name:
                    total_params += hidden_size * hidden_size
                    sub_params += hidden_size * hidden_size
                else:   # key and value proj
                    total_params += hidden_size * hidden_size / layer.self_attn.num_key_value_groups.cpu()
                    sub_params += hidden_size * hidden_size / layer.self_attn.num_key_value_groups.cpu()
            else:
                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size
            if subset[name].bias is not None:
                count += subset[name].bias.data.numel()
                sub_count += subset[name].bias.data.numel()
            
        print(f"layer {i} prune rate {float(sub_count)/sub_params:.6f}")
                
    model.config.use_cache = use_cache 
    return float(count)/total_params

