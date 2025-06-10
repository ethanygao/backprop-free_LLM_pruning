"""
Borrow from 'Moonlit/Compresso'
V1: 1. Tokenize all text 2. Group texts into chunks of block_size (max_seq_length), which means concat a group of samples then chunk into max_seq_length
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math
import logging
from itertools import chain
import transformers
from transformers import DataCollatorWithPadding, default_data_collator
from transformers.testing_utils import CaptureLogger
# from models.tokenization_llama import LlamaTokenizer
from datasets import load_from_disk, load_dataset
import os
import torch

logger = logging.getLogger(__name__)

def get_c4_data_module(model_name, tokenizer, split, max_seq_length=None, max_samples=None, seed=42):
    assert split in ["train", "test"], "argument \'split\' should be in [\"train\", \"test\"]"
    
    # cache tokenized data
    cache_dataset_dir = f"./cache_datasets/{model_name}"; os.makedirs(cache_dataset_dir, exist_ok=True)
    cached_tokenized_data_path = os.path.join(cache_dataset_dir, f"tokenized_c4_{split}.pt")
    
    if os.path.exists(cached_tokenized_data_path):
        logger.info(f"Loading tokenized data from {cached_tokenized_data_path}")
        tokenized_datasets = torch.load(cached_tokenized_data_path)
    else:
        # load samples from c4 dataset
        raw_datasets = load_dataset(path="/data2/zujingliu/workspace/datasets/c4",
                                    # name="c4",
                                    data_files={"train": "en/c4-train.00000-of-01024.json.gz",
                                                "test": "en/c4-validation.00000-of-00008.json.gz"},
                                    split=split
                                    )
        column_names=['timestamp','url','text']
        text_column_name = "text"
        
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
        logger.info(f"Saving tokenized data to {cached_tokenized_data_path}")
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
    
    torch.manual_seed(seed)
    # shuffle the dataset
    dataset = lm_datasets.shuffle(seed)
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))
    
    ##############################################    
    # save the dataset. can be canceled.
    # for iclr25 rebuttal
    # dataset_saved_path = os.path.join(cache_dataset_dir, f"c4_seed{seed}_samples{max_samples}_seqlen{max_seq_length}.pt")
    # if not os.path.exists(dataset_saved_path):
    #     logger.info(f"Saving dataset to {dataset_saved_path}")
    #     torch.save(dataset, dataset_saved_path)
    # else:
    #     logger.info(f"Loading dataset from {dataset_saved_path}")
    #     dataset = torch.load(dataset_saved_path)
    ##############################################
    
    
    return dict(
        # train_dataset=train_dataset,
        # eval_dataset=None,
        dataset=dataset,
        collate_fn=default_data_collator,    # None
        sampler=None,
        pin_memory=True,
        # worker_init_fn=lm_datasets.worker_init_fn,
        shuffle=True
    )
