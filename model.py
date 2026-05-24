import regex as re
import random
import nltk
from collections import Counter, defaultdict

# remove all whitespace and punctuation, and convert to lowercase

def tokenize(text) -> list:
    text = text.strip()  # remove leading and trailing whitespace
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return text.split()


class NGramModel:
    def __init__(self, n, smoothing=False):
        self.n = n
        self.ngrams = Counter()
        self.contexts = defaultdict(Counter)
        self.context_count = Counter()
        self.vocab = set()
        self.smoothing = smoothing

    def train(self, tokens):
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            context = ngram[:-1]
            word = ngram[-1]
            self.ngrams[ngram] += 1
            self.contexts[context][word] += 1
            self.context_count[context] += 1
            self.vocab.add(word)

    def get_probability(self, context, word):
        if self.smoothing:
            return (self.contexts[context][word] + 1) / (
                self.context_count[context] + len(self.vocab)
            )
        else:
            if self.context_count[context] == 0:
                return 0
            return self.contexts[context][word] / self.context_count[context]

    def predict(self, context):
        if context in self.contexts:
            return [
                {word: self.get_probability(context, word)}
                for word in self.contexts[context]
            ]
        return None


def build_model(n=3, smoothing=False):
    nltk.download("gutenberg", quiet=True)
    from nltk.corpus import gutenberg

    text = gutenberg.raw("austen-emma.txt")
    tokens = tokenize(text)
    model = NGramModel(n=n, smoothing=smoothing)
    model.train(tokens)
    return model, tokens
