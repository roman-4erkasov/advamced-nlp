import sys
sys.path.insert(0, "../src")

import re
import os
import datetime
import numpy as np
import torch as th
import pandas as pd
import pickle as pkl
from utils import log
from itertools import islice
from transformers import AutoTokenizer, AutoModelForMaskedLM


def get_date(url):
    dates = re.findall(r"\d\d\d\d\/\d\d\/\d\d", url)
    return next(iter(dates), None)


def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


log("BEGIN")
log("loading prepared data...")
with open("./data/prepared.pkl", "rb") as fp:
    prepared = pkl.load(fp)
vocabulary = prepared["vocabulary"]
texts = prepared["texts"]
contexts = prepared["contexts"]
test_texts = prepared["test_texts"]
y_train = prepared["y_train"]
y_test = prepared["y_test"]
text_train = prepared["texts_train"]
text_test = prepared["texts_test"]

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base").eval().cuda()
batches = batched(text_train, 16)
n_batches = len([b for b in batches])
outputs = []
batches = batched(text_train, 16)
log(f"starting loop over {n_batches=}", ["train"])
for i, batch in enumerate(batches):
    tokens = tokenizer(
        batch,
        return_tensors='pt', 
        max_length=512, 
        truncation=True,
        padding=True,
        pad_to_max_length=True
    )
    input_ids=tokens["input_ids"].cuda()
    attention_mask=tokens["attention_mask"].cuda()
    log(
        f"{input_ids.shape=} {attention_mask.shape=}", 
        ["train", f"{i}/{n_batches}"]
    )
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    out_batch = out.hidden_states[-1].mean(1).detach().cpu().numpy()
    outputs.append(out_batch)
    del batch, tokens,input_ids, attention_mask, out, out_batch
    log(f"{i=}", ["train", f"{i}/{n_batches}"])
output = np.vstack(outputs)
path = "./data/xmlr_train"
log(f"saving train data to \"{path}\"", ["train"])
np.save(path, output)
log("train data saved", ["train"])
batches = batched(text_test, 16)
n_batches = len([b for b in batches])
outputs = []
batches = batched(text_test, 16)
log(f"starting loop over {n_batches=}", ["test"])
for i, batch in enumerate(batches):
    tokens = tokenizer(
        list(batch),
        return_tensors='pt', 
        max_length=512, 
        truncation=True,
        padding=True,
        pad_to_max_length=True
    )    
    input_ids=tokens["input_ids"].cuda()
    attention_mask=tokens["attention_mask"].cuda()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    out_batch = out.hidden_states[-1].mean(1).detach().cpu().numpy()
    outputs.append(out_batch)
    del batch, tokens,input_ids, attention_mask, out, out_batch
    log(f"{i=}", ["test", f"{i}/{n_batches}"])
output = np.vstack(outputs)
path = "./data/xmlr_embs_test"
log(f"saving test data to \"{path}\"", ["test"])
np.save(path, output)
log("test data saved", ["test"])
log("END")

