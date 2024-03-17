import math

import torch
import torch.nn as nn
# from tqdm import tqdm

from .utils import iterate_batches, log


class Tagger(nn.Module):
    def __init__(self, weights_matrix, tagset_size, hidden_dim=300, num_layers=2):
        super().__init__()

        self._embs = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix))

        self._rnn = nn.LSTM(300, hidden_dim, num_layers=num_layers, bidirectional=True)
        self._out = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        embs = self._embs(x)
        out, _ = self._rnn(embs)

        return self._out(out)


def do_epoch(model, criterion, optimizer, sentences, word_to_idx, pos_to_idx, batch_size, name,epoch):
    epoch_loss = 0
    correct = 0
    total = 0

    model.train(True)

    batches_count = math.ceil(len(sentences) / batch_size)

    with torch.autograd.set_grad_enabled(True):
        # with tqdm(total=batches_count) as progress_bar:
        for i, (x, y) in enumerate(iterate_batches(sentences, word_to_idx, pos_to_idx, batch_size)):
            x, y = torch.LongTensor(x), torch.LongTensor(y)

            logits = model(x)

            loss = criterion(logits.view(-1, logits.shape[-1]), y.view(-1))
            epoch_loss += loss.item()

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pred = torch.argmax(logits, dim=-1)
            mask = (y != 0).float()
            cur_correct = ((pred == y).float() * mask).sum()
            cur_total = mask.sum().item()

            correct += cur_correct
            total += cur_total

            # progress_bar.update()
            # progress_bar.set_description('{:>5s} Loss = {:.5f}, Accuracy = {:.2%}'.format(
            #     name, loss.item(), cur_correct / cur_total)
            # )
            log(f"acc={cur_correct / cur_total} loss={loss.item()}", headers=[f"e{epoch}",f"{i}/{batches_count}"])

        # progress_bar.set_description('{:>5s} Loss = {:.5f}, Accuracy = {:.2%}'.format(
        #     name, epoch_loss / batches_count, correct / total)
        # )
        log(f"train finished: acc={correct / total} loss={epoch_loss / batches_count}", headers=[f"e{epoch}"])

    return epoch_loss / batches_count, correct / total


def fit(model, criterion, optimizer, sentences, word_to_idx, pos_to_idx, batch_size=32, epochs_count=1):
    for epoch in range(epochs_count):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
        train_loss, train_acc = do_epoch(model, criterion, optimizer,
                                         sentences, word_to_idx, pos_to_idx,
                                         batch_size, name_prefix + 'Train:',epoch)


def score_model(model, data, word_to_idx, pos_to_idx, batch_size=128):
    model.eval()

    correct, total = 0, 0

    for x, y in iterate_batches(data, word_to_idx, pos_to_idx, batch_size):
        x, y = torch.LongTensor(x), torch.LongTensor(y)

        logits = model(x)

        pred = torch.argmax(logits, dim=-1)
        mask = (y != 0).float()
        cur_correct = ((pred == y).float() * mask).sum()
        cur_total = mask.sum().item()

        correct += cur_correct
        total += cur_total

    print('Accuracy = {:.2%}'.format(correct / total))
