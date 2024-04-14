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
FNAME, _ = os.path.splitext(os.path.basename(__file__))
RUN_ID = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_freqs(texts, vcabulary):
    word_qty = Counter(w for txt in texts for w in txt)
    word2freq = np.zeros(vocabulary.size)
    for idx in range(vocabulary.size):
        word = vocabulary.get_word(idx)
        qty = word_qty[word]
        word2freq[idx] = qty
    word2freq /= word2freq.sum()
    return word2freq


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


class CBOWNS(LightningModule):
    def __init__(self, vocab_size, embedding_dim=128, sample_sz=4, word2freq=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.sample_sz = sample_sz
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_layer = nn.Linear(embedding_dim, vocab_size)
        self.loss = nn.CrossEntropyLoss()
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
        if word2freq is None:
            self.word2freq = torch.ones(self.vocab_size)
        else:
            self.word2freq = torch.from_numpy(word2freq)

    def forward(self, centrals, contexts):
        centrals = centrals.to(DEVICE)
        contexts = contexts.to(DEVICE)
        batch_sz = centrals.shape[0]
        projections = self.embeddings.forward(contexts).sum(axis=1)
        logits = self.out_layer.forward(projections)
        centrals_ohe = torch.nn.functional.one_hot(centrals, self.vocab_size)
        out_loss = torch.bmm(
            centrals_ohe.float(), 
            logits.unsqueeze(2)
        ).sum(1).sigmoid().log().squeeze()

        noise_inp = torch.multinomial(
            self.word2freq, batch_sz*self.sample_sz, replacement=True
        ).view(batch_sz, self.sample_sz).to(DEVICE)
        noise_proj = self.embeddings.forward(noise_inp).sum(axis=1)
        noise_logits = self.out_layer.forward(noise_proj)
        noise_loss = torch.bmm(
            centrals_ohe.float(), 
            noise_logits.neg().unsqueeze(2)
        ).sigmoid().log().squeeze()#.sum(1)

        return -(out_loss + noise_loss).mean()
    
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
model = CBOWNS(vocabulary.size).to(DEVICE)
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
    filename=f"{FNAME}-{{epoch}}-{{val_loss:.2f}}",
    save_top_k=3,
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
    accelerator="gpu", devices="auto"
    # enable_progress_bar=False,
)
log("training...")
trainer.fit(model, train_loader, val_loader)
log("END")
