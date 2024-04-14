import random
import os
import datetime
import numpy as np
import torch
import __main__


def log(msg:str, headers=None):
    dttm = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # if "__file__" not in globals() and "__file__" not in locals():
    if not hasattr(__main__, "__file__"):
        script = "jupyter"
    else:
        script = os.path.basename(__main__.__file__)
    if headers is None:
        headers = []
    header_line = f"[{dttm}][{script}]" + "".join(f"[{h}]" for h in headers)
    print(f"{header_line} {msg}")


def get_next_batch(
    contexts, 
    window_size, 
    batch_size, 
    epochs_count,
):
    assert batch_size % (window_size * 2) == 0
    central_words, contexts = zip(*contexts)
    batch_size //= (window_size * 2)
    for epoch in range(epochs_count):
        indices = np.arange(len(contexts))
        np.random.shuffle(indices)
        batch_begin = 0
        while batch_begin < len(contexts):
            batch_indices = indices[batch_begin: batch_begin + batch_size]
            batch_contexts, batch_centrals = [], []
            for data_ind in batch_indices:
                central_word, context = central_words[data_ind], contexts[data_ind]
                batch_contexts.extend(context)
                batch_centrals.extend([central_word] * len(context))
            batch_begin += batch_size
            yield (
                torch.LongTensor(batch_contexts), 
                torch.LongTensor(batch_centrals)
            )

# print(next(get_next_batch(contexts, window_size=2, batch_size=64, epochs_count=10)))