from typing import Union

import numpy
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import PAIRED_DISTANCES
from sklearn.utils.validation import (check_array,
                                      check_consistent_length,
                                      check_is_fitted,
                                      check_scalar,
                                      column_or_1d)


class SimilarItemsContentRecommender(BaseEstimator):
    def __init__(self, distance_metric: str = 'euclidean') -> None:
        super().__init__()
        self.distance_metric = distance_metric

    def fit(self,
            X: sparse.csr_matrix,
            y: numpy.ndarray) -> 'SimilarItemsContentRecommender':
        self.item_ids = check_array(y, dtype=None, ensure_2d=False)
        self.item_embeddings: sparse.csr_matrix = check_array(X, accept_sparse=True)
        check_consistent_length(self.item_embeddings, self.item_ids)

        distance_metric = check_scalar(self.distance_metric,
                                       name='item-item nearest neighbour model`s type (implicit parameter)',
                                       target_type=str)
        if distance_metric not in PAIRED_DISTANCES:
            raise ValueError(distance_metric)

        self.similarities = pairwise_distances(self.item_embeddings, metric=distance_metric)

        self.is_fitted_ = True
        return self

    def predict(self, X: numpy.ndarray, k: int) -> numpy.ndarray:
        check_is_fitted(self, 'is_fitted_')
        input_ids = column_or_1d(check_array(X, dtype=numpy.object_, ensure_2d=False))
        k = check_scalar(k,
                         name='output recommendations count',
                         target_type=int,
                         min_val=1,
                         max_val=len(self.item_ids))

        preds = []
        for item_id in input_ids:
            input_idx = numpy.where(self.item_ids == item_id)[0][0]
            output_idxs = numpy.argsort(self.similarities[input_idx])[:k+1]

            preds.append([self.item_ids[idx] for idx in output_idxs if idx != input_idx])

        return numpy.array(preds, dtype=numpy.object_)
