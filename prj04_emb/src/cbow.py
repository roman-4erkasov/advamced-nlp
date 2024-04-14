import sys
sys.path.append("../src/")
import os
import time
import json
import torch
import random
import datetime
import pickle as pkl
import torch.nn as nn
from types import NoneType
from itertools import cycle
import torch.optim as optim
from utils import get_next_batch
from vocabulary import Vocabulary
from pytorch_lightning import Trainer
from typing import Union, Mapping, Any
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint


DEBUG = False
BATCH_SIZE = 256
EPOCHS = 10
RUN_ID = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
FNAME, _ = os.path.splitext(os.path.basename(__file__))


def log(msg: str):
    now = datetime.datetime.now()
    dttm = now.strftime(format="%Y-%m-%d %H:%M:%S.%f") 
    print(f"[{dttm}] {msg}")


def get_samples(tokenized_texts, window_size, texts_count):
    for text_num, tokens in enumerate(tokenized_texts):
        if texts_count and text_num >= texts_count:
            break
        for i in range(len(tokens)):
            central_word = torch.LongTensor(
                [vocabulary.get_index(tokens[i])]
            )
            context = torch.LongTensor(
                [
                    vocabulary.get_index(tokens[i + delta])
                    for delta in range(-window_size, window_size + 1)
                    if 0 <= (i + delta) < len(tokens)
                ]
            )
            if 2*window_size == context.shape[0]:
                yield central_word, context


def get_samples_cycle(tokenized_texts, window_size, texts_count):
    while True:
        for sample in get_samples(tokenized_texts, window_size, texts_count):
            yield sample


class Word2VecDataset(Dataset):
    def __init__(self, tokenized_texts, vocabulary, window_size=2, texts_count=100000):
        self.samples = list(get_samples(tokenized_texts, window_size, texts_count))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index]


class Word2VecIterableDataset(IterableDataset):
    def __init__(self, tokenized_texts, vocabulary, window_size=2, texts_count=None):
        self.tokenized_texts = tokenized_texts
        self.vocabulary = vocabulary
        self.window_size = window_size
        self.texts_count = texts_count

    def __iter__(self):
        return get_samples_cycle(self.tokenized_texts, self.window_size, self.texts_count)


class CBOWModel(LightningModule):
    def __init__(self, vocab_size, embedding_dim=128):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_layer = nn.Linear(embedding_dim, vocab_size)
        self.loss = nn.CrossEntropyLoss()
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
    
    def forward(self, centrals, contexts):
        projections = self.embeddings.forward(contexts).sum(axis=1)
        logits = self.out_layer.forward(projections)
        loss = self.loss(logits, centrals.squeeze())
        return loss
    
    def training_step(self, batch, batch_nb):
        result = self(*batch)
        self.log("loss", result)
        return {'loss': result}
    
    def validation_step(self, batch, batch_nb):
        result = self(*batch)
        self.log("val_loss", result)  
        return {'val_loss': result}

    def test_step(self, batch, batch_nb):
        result = self(*batch)
        self.log("test_loss", result)
        return {'test_loss': self(*batch)}

    def on_train_batch_end(
        self,
        outputs: Union[torch.Tensor, Mapping[str, Any], NoneType],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.train_outputs.append(outputs)
    
    def on_train_epoch_end(self):
        outputs = self.train_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'loss': avg_loss}
        self.log("train_loss_epoch", avg_loss, on_step=False, on_epoch=True)
        return {'train_loss_epoch': avg_loss, 'progress_bar': tensorboard_logs}
    
    def on_validation_batch_end(
        self,
        outputs: Union[torch.Tensor, Mapping[str, Any], NoneType],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.val_outputs.append(outputs)
    
    def on_validation_epoch_end(self):
        outputs = self.val_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log("val_loss_epoch", avg_loss, on_step=False, on_epoch=True)
        return {'val_loss_epoch': avg_loss, 'progress_bar': tensorboard_logs}

    def on_test_batch_end(
        self,
        outputs: Union[torch.Tensor, Mapping[str, Any], NoneType],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.test_outputs.append(outputs)
    
    def on_test_epoch_end(self):
        outputs = self.test_outputs 
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        self.log("test_loss_epoch", avg_loss, on_step=False, on_epoch=True)
        return {'test_loss_epoch': avg_loss, 'progress_bar': tensorboard_logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return [optimizer]


log("BEGIN")
log("loading prepared data...")
with open("./data/prepared.pkl", "rb") as fp:
    prepared = pkl.load(fp)
vocabulary = prepared["vocabulary"]
texts = prepared["texts"]
contexts = prepared["contexts"]
test_texts = prepared["test_texts"]
del prepared
log("data loaded")
random.shuffle(texts)
train_data = Word2VecIterableDataset(texts, vocabulary)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
random.shuffle(test_texts)
val_data = Word2VecIterableDataset(test_texts, vocabulary)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
model = CBOWModel(vocabulary.size)
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0,
    patience=5,
    verbose=True,
    mode="min",
)
ckpt_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=f"./lightning_logs/[{FNAME}][{RUN_ID}]/ckpt",
    filename=f"{{epoch}}-{{val_loss:.2f}}",
    save_top_k=4,
    mode="min",
    save_last=True
)
trainer = Trainer(
    max_epochs=EPOCHS,
    callbacks=[
        # early_stop_callback, 
        ckpt_callback
    ],
    limit_train_batches=2 if DEBUG else 40000,
    limit_val_batches=2 if DEBUG else 500,
    val_check_interval=1 if DEBUG else 2000,
    logger=CSVLogger(save_dir=f"./lightning_logs/[{FNAME}][{RUN_ID}]/logs"),
    # enable_progress_bar=False,
)
log("training...")
trainer.fit(model, train_loader, val_loader)
log("END")
