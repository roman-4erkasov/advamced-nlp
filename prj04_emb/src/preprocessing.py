import pandas as pd
import re
import pickle as pkl
import datetime as dt
from razdel import tokenize, sentenize
from string import punctuation
from vocabulary import Vocabulary


def get_date(url):
    dates = re.findall(r"\d\d\d\d\/\d\d\/\d\d", url)
    return next(iter(dates), None)


def get_texts(dataset):
    texts = []
    for text in dataset["text"]:
        for sentence in sentenize(text):
            texts.append([token.text.lower() for token in tokenize(sentence.text) if token.text not in punctuation])

    for title in dataset["title"]:
        texts.append([token.text.lower() for token in tokenize(title) if token.text not in punctuation])
    return texts


def build_contexts(tokenized_texts, vocabulary, window_size):
    contexts = []
    for tokens in tokenized_texts:
        for i in range(len(tokens)):
            central_word = vocabulary.get_index(tokens[i])
            context = [vocabulary.get_index(tokens[i + delta]) for delta in range(-window_size, window_size + 1)
                       if delta != 0 and i + delta >= 0 and i + delta < len(tokens)]
            if len(context) != 2 * window_size:
                continue
            contexts.append((central_word, context))
    return contexts


dataset = pd.read_csv("../lenta-ru-news.csv", sep=',', quotechar='\"', escapechar='\\', encoding='utf-8', header=0)
dataset["date"] = dataset["url"].apply(lambda x: dt.datetime.strptime(get_date(x), "%Y/%m/%d"))
dataset = dataset[dataset["date"] > "2017-01-01"]
dataset["text"] = dataset["text"].apply(lambda x: x.replace("\xa0", " "))
dataset["title"] = dataset["title"].apply(lambda x: x.replace("\xa0", " "))
train_dataset = dataset[dataset["date"] < "2018-04-01"]
test_dataset = dataset[dataset["date"] > "2018-04-01"]

texts = get_texts(train_dataset)
test_texts = get_texts(test_dataset)

assert len(texts) == 827217
assert len(texts[0]) > 0
assert texts[0][0].islower()
print(texts[0])

vocabulary = Vocabulary()
vocabulary.build(texts)
assert vocabulary.word2index[vocabulary.index2word[10]] == 10
print(vocabulary.size)
print(vocabulary.top(100))

contexts = build_contexts(texts, vocabulary, window_size=2)
print(contexts[:5])
print(vocabulary.get_word(contexts[0][0]), [vocabulary.get_word(index) for index in contexts[0][1]])

with open("../data/prepared.pkl", "wb") as fp:
    pkl.dump(
        file=fp, 
        obj={
            "vocabulary": vocabulary,
            "texts": texts,
            "contexts": contexts,
        }
    )

