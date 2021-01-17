import string

from pyonmttok import Tokenizer
from pymorphy2 import MorphAnalyzer

tokenizer = Tokenizer("conservative", joiner_annotate=False)
morph = MorphAnalyzer()


def tokenize(text, lower=True):
    text = str(text).strip().replace("\n", " ").replace("\xa0", " ")
    if lower:
        text = text.lower()
    tokens, _ = tokenizer.tokenize(text)
    return tokens


def tokenize_to_lemmas(text):
    tokens = tokenize(text)
    tokens = filter(lambda x: x not in string.punctuation, tokens)
    tokens = filter(lambda x: not x.isnumeric(), tokens)
    tokens = filter(lambda x: len(x) >= 2, tokens)
    tokens = [morph.parse(token)[0].normal_form for token in tokens]
    return tokens
