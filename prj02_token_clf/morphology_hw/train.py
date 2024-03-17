import torch
import torch.nn as nn
import torch.optim as optim

from src import utils
from src import model


train = utils.get_sentence('data/train.csv')
# Create supplementary stuff
pos_to_idx, idx_to_pos = utils.get_tag_mapping(train)
word_to_idx, idx_to_word, embeddings = utils.get_embeddings(train, 'data/model.txt')
# Split data into train and test
test = train[-1000:]
train = train[:-1000]
# Configure the model
tagger = model.Tagger(weights_matrix=embeddings, tagset_size=len(pos_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(tagger.parameters())
# Fit the model
model.fit(tagger, criterion, optimizer, train, word_to_idx, pos_to_idx, batch_size=32, epochs_count=20)
# Score the model
score = model.score_model(tagger, test, word_to_idx, pos_to_idx)
print(f"mode score: {score}")
torch.save(f="tagger.pth", obj=tagger)