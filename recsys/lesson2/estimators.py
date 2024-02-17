from functools import partial

import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer

from utils.features.extraction import Word2VecVectorizer
from utils.features.preprocessing import function_1d_wrapper
from utils.text.preprocessing import document_tokenize


def simple_vectorizer() -> CountVectorizer:
    vectorizer = CountVectorizer(
        lowercase=False,
        tokenizer=lambda items: items if isinstance(items, numpy.ndarray) else [],
        token_pattern=None
    )

    return Pipeline(
        [('vectorizer', vectorizer),
         ('normalizer', Normalizer())]
    )


def embedding_vectorizer() -> Pipeline:
    vectorizer = Word2VecVectorizer()

    return Pipeline(
        [('tokenizer', FunctionTransformer(partial(function_1d_wrapper,
                                                   func1d=document_tokenize,
                                                   dtype=numpy.object_))),
         ('embedding', vectorizer),
         ('normalizer', Normalizer())]
    )
