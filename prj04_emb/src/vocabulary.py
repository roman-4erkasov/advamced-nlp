from collections import Counter


class Vocabulary:
    def __init__(self):
        self.word2index = {"<unk>": 0}
        self.index2word = ["<unk>"]

    def build(self, texts, min_count=10):
        words_counter = Counter(token for tokens in texts for token in tokens)
        for word, count in words_counter.most_common():
            if count >= min_count:
                self.word2index[word] = len(self.word2index)
        self.index2word = [word for word, _ in sorted(self.word2index.items(), key=lambda x: x[1])]

    @property
    def size(self):
        return len(self.index2word)

    def top(self, n=100):
        return self.index2word[1:n+1]

    def get_index(self, word):
        return self.word2index.get(word, 0)

    def get_word(self, index):
        return self.index2word[index]
