# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from pprint import pprint

from . import c4, wikitext

TASK_EVALUATE_REGISTRY = {
    # "c4": c4.evaluate_c4,
    # "alpaca-gpt4": alpaca.evaluate_alpaca,
    # "alpaca-cleaned": alpaca.evaluate_alpaca,
}


TASK_DATA_MODULE_REGISTRY = {
    "c4": c4.get_c4_data_module,
    "wikitext2": wikitext.get_wikitext_data_module,
}


def get_task_evaluater(task_name):
    if task_name not in TASK_EVALUATE_REGISTRY:
        print("Available tasks:")
        pprint(TASK_EVALUATE_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
    
    return TASK_EVALUATE_REGISTRY[task_name]


def get_data_module(task_name):
    if task_name not in TASK_DATA_MODULE_REGISTRY:
        print("Available tasks:")
        pprint(TASK_DATA_MODULE_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
    
    return TASK_DATA_MODULE_REGISTRY[task_name]

'''
GPU uitls
'''
from typing import TypeVar
import torch

def sync_gpus() -> None:
    """Sync all GPUs to make sure all operations are finished, needed for correct benchmarking of latency/throughput."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)

T = TypeVar('T')
def map_tensors(obj: T, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> T:
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
    else:
        return obj