"""
Borrow from 'Moonlit/Compresso'
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from datasets import load_dataset
from itertools import chain
import transformers
from transformers import DataCollatorWithPadding, default_data_collator, is_torch_tpu_available
from datasets import load_dataset, Dataset
import evaluate
from transformers.testing_utils import CaptureLogger
import os
import torch

logger = logging.getLogger(__name__)

def get_wikitext_data_module_v1(model_name, tokenizer, split, max_seq_length=None, max_samples=None, seed=42):
    assert split in ["train", "test"], "argument \'split\' should be in [\"train\", \"test\"]"
    
    # cache tokenized data
    cache_dataset_dir = f"./cache_datasets/{model_name}"; os.makedirs(cache_dataset_dir, exist_ok=True)
    cached_tokenized_data_path = os.path.join(cache_dataset_dir, f"tokenized_wikitext2_{split}.pt")
    
    if os.path.exists(cached_tokenized_data_path):
        tokenized_datasets = torch.load(cached_tokenized_data_path)
    else:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "/data2/zujingliu/workspace/datasets/wikitext",        # 'wikitext',
            'wikitext-2-raw-v1',
            split=split
        )
        # if split == "train":
        #     column_names = raw_datasets["train"].column_names
        # else:
        #     column_names = raw_datasets["test"].column_names
        column_names = raw_datasets.column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return output

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=32,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        
        torch.save(tokenized_datasets, cached_tokenized_data_path)
    
    if max_seq_length is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(max_seq_length, tokenizer.model_max_length)
        
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=32,
        load_from_cache_file=False,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    # lm_datasets_eval = tokenized_datasets.map(
    #     group_texts,
    #     batched=True,
    #     num_proc=32,
    #     load_from_cache_file=False,
    #     desc=f"Grouping texts in chunks of {block_size}",
    # )
    
    ###############################################
    # train_dataset = None
    # if split == "train":
    #     # if "train" not in tokenized_datasets:
    #     #     raise ValueError("split=\"train\" requires a train dataset")
    #     # train_dataset = lm_datasets["train"]
    #     train_dataset = lm_datasets
    #     if max_samples is not None:
    #         train_dataset = train_dataset.select(range(max_samples))
    
    # eval_dataset = None
    # if split == "test":
    #     # if "validation" not in tokenized_datasets:
    #     #     raise ValueError("split=\"test\" requires a test dataset")
    #     # eval_dataset = lm_datasets_eval["test"]
    #     # eval_dataset = lm_datasets["test"]
    #     eval_dataset = lm_datasets
    #     if max_samples is not None:
    #         eval_dataset = eval_dataset.select(range(max_samples))
    ###############################################
    
    torch.manual_seed(seed)
    # shuffle the dataset
    dataset = lm_datasets.shuffle(seed)
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))
            
    return dict(
        # train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        dataset=dataset,
        collate_fn=default_data_collator,    # None
        sampler=None,
        pin_memory=True,
        # worker_init_fn=lm_datasets.worker_init_fn,
        shuffle=True
    )
        
def get_wikitext_data_module(model_name, tokenizer, split, max_seq_length=None, max_samples=None, seed=42):
    assert split in ["train", "test"], "argument \'split\' should be in [\"train\", \"test\"]"
    
    # cache tokenized data
    cache_dataset_dir = f"./cache_datasets/{model_name}"; os.makedirs(cache_dataset_dir, exist_ok=True)
    cached_tokenized_data_path = os.path.join(cache_dataset_dir, f"tokenized_wikitext2_{split}.pt")
    
    if os.path.exists(cached_tokenized_data_path):
        tokenized_data = torch.load(cached_tokenized_data_path)
    else:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "/data2/zujingliu/workspace/datasets/wikitext",        # 'wikitext',
            'wikitext-2-raw-v1',
            split=split
        )

        column_names = raw_datasets.column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        if split == "train":
            # use " " to concatenate the text
            tokenized_data = tokenizer(" ".join(raw_datasets['text']), return_tensors='pt')
        elif split == "test":
            # use "\n\n" to concatenate the text
            tokenized_data = tokenizer("\n\n".join(raw_datasets['text']), return_tensors='pt')
        else:
            raise ValueError(f"split={split} is not supported")
        
        # cache the tokenized data
        torch.save(tokenized_data, cached_tokenized_data_path)
        
    if max_seq_length is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(max_seq_length, tokenizer.model_max_length)
    
    print("stop here")
    ## construct the dataset, BEGIN ##
    input_ids = tokenized_data['input_ids'].squeeze()
    attention_mask = tokenized_data['attention_mask'].squeeze()
    
    num_chunks = input_ids.size(0) // block_size
    input_ids = input_ids[:num_chunks * block_size].view(-1, block_size)
    attention_mask = attention_mask[:num_chunks * block_size].view(-1, block_size)
    
    tokenized_dataset = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}
    lm_datasets = Dataset.from_dict(tokenized_dataset)
    ## construct the dataset, END ##
    
    
    torch.manual_seed(seed)
    # shuffle the dataset
    dataset = lm_datasets.shuffle(seed)
    if max_samples is not None:
        if max_samples > len(dataset):
            max_samples = len(dataset)
        dataset = dataset.select(range(max_samples))
            
    return dict(
        # train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        dataset=dataset,
        collate_fn=default_data_collator,    # None
        sampler=None,
        pin_memory=True,
        # worker_init_fn=lm_datasets.worker_init_fn,
        shuffle=True
    )