import pandas as pd
import re
from utils import log
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
            context = [
                vocabulary.get_index(tokens[i + delta]) 
                for delta in range(-window_size, window_size + 1)
                if delta != 0 and i + delta >= 0 and i + delta < len(tokens)
            ]
            if len(context) != 2 * window_size:
                continue
            contexts.append((central_word, context))
    return contexts


log("BEGIN")
log("reading dataset...")
dataset = pd.read_csv("./data/lenta-ru-news.csv", sep=',', quotechar='\"', escapechar='\\', encoding='utf-8', header=0)
log("preparing dataset formats...")
dataset["date"] = dataset["url"].apply(lambda x: dt.datetime.strptime(get_date(x), "%Y/%m/%d"))
dataset = dataset[dataset["date"] > "2017-01-01"]
dataset["text"] = dataset["text"].apply(lambda x: x.replace("\xa0", " "))
dataset["title"] = dataset["title"].apply(lambda x: x.replace("\xa0", " "))
log("splitting dataset...")
train_dataset = dataset[dataset["date"] < "2018-04-01"]
test_dataset = dataset[dataset["date"] > "2018-04-01"]
log("extracting tokens...")
texts = get_texts(train_dataset)
test_texts = get_texts(test_dataset)
assert len(texts) == 827217
assert len(texts[0]) > 0
assert texts[0][0].islower()
log(f"example: {texts[0]=}")
log("building vocabulary...")
vocabulary = Vocabulary()
vocabulary.build(texts)
assert vocabulary.word2index[vocabulary.index2word[10]] == 10
log(f"{vocabulary.size=}")
log(f"{vocabulary.top(10)=}")
log("building context...")
contexts = build_contexts(texts, vocabulary, window_size=2)
log(f"{contexts[:5]=}")
log("extracting targets...")
target_labels = set(train_dataset["topic"].dropna().tolist())
target_labels -= {"69-я параллель", "Крым", "Культпросвет ", "Оружие", "Бизнес", "Путешествия"}
target_labels = list(target_labels)
log(f"{target_labels=}")
pattern = r'(\b{}\b)'.format('|'.join(target_labels))
train_with_topics = train_dataset[train_dataset["topic"].str.contains(pattern, case=False, na=False)]
# train_with_topics = train_with_topics.head(20_000)
test_with_topics = test_dataset[test_dataset["topic"].str.contains(pattern, case=False, na=False)]
y_train = train_with_topics["topic"].apply(lambda x: target_labels.index(x)).to_numpy()
y_test = test_with_topics["topic"].apply(lambda x: target_labels.index(x)).to_numpy()
path = "./data/prepared.pkl"
log(f"saving data to \"{path}\"...")
with open(path, "wb") as fp:
    pkl.dump(
        file=fp, 
        obj={
            "vocabulary": vocabulary,
            "texts": texts,
            "contexts": contexts,
            "test_texts": test_texts,
            "y_train": y_train,
            "y_test": y_test,
            "target_labels": target_labels,
            "texts_test": test_with_topics["text"],
            "texts_train": train_with_topics["text"]
        }
    )
log("END")
