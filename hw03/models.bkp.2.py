
# Models for word alignment
from collections import defaultdict

class TranslationModel:
    "Models conditional distribution over trg words given a src word, i.e. t(f|e)."

    def __init__(self, src_corpus, trg_corpus):
        self._src_trg_counts = defaultdict(lambda: defaultdict(float))
        self._trg_given_src_probs = {}

    def get_conditional_prob(self, src_token, trg_token):
        "Return the conditional probability of trg_token given src_token, i.e. t(f|e)."
        if src_token not in self._trg_given_src_probs:
            return 1.0
        if trg_token not in self._trg_given_src_probs[src_token]:
            return 1.0
        return self._trg_given_src_probs[src_token][trg_token]

    def collect_statistics(self, src_tokens, trg_tokens, posterior_matrix):
        "Accumulate counts of translations from matrix: matrix[j][i] = p(a_j=i|e, f)"
        assert len(posterior_matrix) == len(trg_tokens)
        for posterior in posterior_matrix:
            assert len(posterior) == len(src_tokens)

        for i, src_token in enumerate(src_tokens):
            for j, trg_token in enumerate(trg_tokens):
                self._src_trg_counts[src_token][trg_token] += posterior_matrix[j][i]

    def recompute_parameters(self):
        "Reestimate parameters and reset counters."
        self._trg_given_src_probs = {}
        for src_token in self._src_trg_counts:
            self._trg_given_src_probs[src_token] = {}
            trg_tokens_dict = self._src_trg_counts[src_token]
            trg_tokens_sum = sum(trg_tokens_dict.values())
            for trg_token in trg_tokens_dict:
                self._trg_given_src_probs[src_token][trg_token] = \
                    1.0 * trg_tokens_dict[trg_token] / trg_tokens_sum

class PriorModel:
    "Models the prior probability of an alignment given only the sentence lengths and token indices."

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for more sophisticated models."
        self._distance_counts = defaultdict(lambda: defaultdict(float))
        self._distance_probs = {}

    def get_prior_prob(self, src_index, trg_index, src_length, trg_length):
        "Returns a uniform prior probability."
        if src_index not in self._distance_probs:
            return 1.0 / src_length
        if trg_index not in self._distance_probs[src_index]:
            return 1.0 / trg_length
        return self._distance_probs[src_index][trg_index]

    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        "Count the necessary statistics from this matrix if needed."
        assert len(posterior_matrix) == trg_length
        for posterior in posterior_matrix:
            assert len(posterior) == src_length

        for i in range(src_length):
            for j in range(trg_length):
                self._distance_counts[i][j] += posterior_matrix[j][i]

    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        self._distance_probs = {}
        for src_length in self._distance_counts:
            self._distance_probs[src_length] = {}
            trg_length_dict = self._distance_counts[src_length]
            trg_length_sum = sum(trg_length_dict.values())
            for trg_length in trg_length_dict:
                self._distance_probs[src_length][trg_length] = (
                    1.0 * trg_length_dict[trg_length] 
                    / 
                    trg_length_sum
                )
