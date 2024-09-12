import os
import numpy as np
import torch
import json
import datasets
from datasets import load_dataset, concatenate_datasets
import random
from itertools import chain
from transformers import DataCollatorForLanguageModeling, default_data_collator
import transformers
import tqdm

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_qat_dataset(name, tokenizer):
    if name == "wikitext2":
        data, data_collator = get_wikitext2_train(tokenizer=tokenizer)
    elif name == "c4":
        data, data_collator = get_c4_train(tokenizer=tokenizer)
    elif name == "c4_wiki":
        data, data_collator = get_c4_wiki_train(tokenizer=tokenizer)
    return data, data_collator

def get_eval_loaders(name, tokenizer):
    if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
                print(f"bos/eos tokens updated: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
            except AttributeError:
                pass
                print(f"bos/eos tokens unchanged: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(tokenizer)
        return get_ptb(tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(tokenizer)
        return get_c4(tokenizer)


def get_wikitext2_train(tokenizer, seed=0, seqlen=2048):
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )

    wikitext_dataset = datasets.Dataset.from_dict(
            {
                "text": [
                    # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                    "\n\n".join(dataset["text"])
                ],
            },
        )

    # Hacks to get around the `remove_columns` to be used later.
    wikitext_dataset = (
        wikitext_dataset  # type: ignore
        .add_column(
            name="timestamp",
            column=wikitext_dataset["text"])
        .add_column(
            name="url",
            column=wikitext_dataset["text"])
    )
    column_names = list(wikitext_dataset.features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = wikitext_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    block_size = 2048

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

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
    )
    return processed_dataset, default_data_collator



def get_c4_train(tokenizer, seed=0, seqlen=2048):
    raw_datasets = load_dataset(
        "allenai/c4",
        #"allenai--c4",
        data_files={
            "train": "en/c4-train.00000-of-01024.json.gz",
            "validation": "en/c4-validation.00000-of-00008.json.gz",
        },
    )
    _wikitext_dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
        )
    # Hacks to be consistent with other works' preprocessing.
    wikitext_dataset = datasets.Dataset.from_dict(
        {
            "text": [
                # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                "\n\n".join(_wikitext_dataset["text"])
            ],
        },
    )

    wikitext_dataset = (
        wikitext_dataset  # type: ignore
        .add_column(
            name="timestamp",
            column=wikitext_dataset["text"])
        .add_column(
            name="url",
            column=wikitext_dataset["text"])
    )

    raw_datasets["wikitext"] = wikitext_dataset

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    block_size = 2048

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

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
    )
    return processed_dataset["train"], default_data_collator



def get_c4_wiki_train(tokenizer, seed=0, seqlen=2048):
    raw_datasets = load_dataset(
        "allenai/c4",
        #"allenai--c4",
        data_files={
            "train": "en/c4-train.00000-of-01024.json.gz",
            "validation": "en/c4-validation.00000-of-00008.json.gz",
        },
    )
    _wikitext_dataset_train = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
        )
    _wikitext_dataset_eval = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
        )
    # Hacks to be consistent with other works' preprocessing.
    wikitext_dataset_train = datasets.Dataset.from_dict(
        {
            "text": [
                # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                "\n\n".join(_wikitext_dataset_train["text"])
            ],
        },
    )
    wikitext_dataset_eval = datasets.Dataset.from_dict(
        {
            "text": [
                # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                "\n\n".join(_wikitext_dataset_eval["text"])
            ],
        },
    )

    wikitext_dataset_train = (
        wikitext_dataset_train  # type: ignore
        .add_column(
            name="timestamp",
            column=[None for _ in range(len(wikitext_dataset_train["text"]))])
        .add_column(
            name="url",
            column=wikitext_dataset_train["text"])
    )
    wikitext_dataset_eval = (
        wikitext_dataset_eval  # type: ignore
        .add_column(
            name="timestamp",
            column=wikitext_dataset_eval["text"])
        .add_column(
            name="url",
            column=wikitext_dataset_eval["text"])
    )

    raw_datasets["train"] = concatenate_datasets([
        raw_datasets["train"],
        wikitext_dataset_train])
    raw_datasets["wikitext"] = wikitext_dataset_eval

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    block_size = 2048

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

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
    )
    return processed_dataset["train"], default_data_collator


def get_wikitext2(tokenizer, seqlen=2048):
    testdata = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    return testenc

def get_ptb(tokenizer, seqlen=2048):
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    return testenc

def get_c4(tokenizer, seqlen=2048):
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return valenc 

def get_ptb_new(tokenizer, seqlen=2048):
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    return testenc

def get_c4_new(tokenizer, seqlen=2048):
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return valenc






