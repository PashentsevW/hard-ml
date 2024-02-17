from typing import Optional

import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import (check_array,
                                      check_scalar,
                                      column_or_1d)

from ...text.models import pretrained_models


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word_model_id: Optional[str] = None) -> None:
        super().__init__()
        self.word_model_id = word_model_id

    def fit(self, X, y = None) -> 'Word2VecVectorizer':
        check_scalar(self.word_model_id,
                     name='pretrained word model id',
                     target_type=str)
        if self.word_model_id not in pretrained_models:
            raise KeyError(self.word_model_id)

        return self        

    def transform(self, X: numpy.ndarray) -> numpy.ndarray:
        documents = column_or_1d(check_array(X, dtype=numpy.object_, ensure_2d=False))

        word_model = pretrained_models[self.word_model_id]

        tfidf_model = TfidfVectorizer(lowercase=False,
                                      tokenizer=lambda items: items,
                                      token_pattern=None)
        token_weights = tfidf_model.fit_transform(documents)

        Xt = []
        for idx in range(len(documents)):
            vectors = numpy.array([word_model[token]
                                   for token in documents[idx]
                                   if token in word_model])

            if vectors.shape[0] == 0:
                Xt.append(numpy.full(shape=word_model.vector_size, fill_value=0))
                continue
            
            weights = numpy.array([token_weights.getrow(idx).A.flatten()[tfidf_model.vocabulary_[token]]
                                   for token in documents[idx]
                                   if token in word_model])
            Xt.append((weights.reshape(-1, 1) * vectors).sum(axis=0) / weights.sum())

        return numpy.array(Xt)
