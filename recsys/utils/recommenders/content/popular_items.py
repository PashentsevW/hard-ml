import numpy
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (check_array,
                                      check_consistent_length,
                                      check_is_fitted,
                                      check_scalar,
                                      column_or_1d)


class PopularItemsContentRecommender(BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: numpy.ndarray, y: numpy.ndarray) -> 'PopularItemsContentRecommender':
        item_ids = check_array(y, dtype=None, ensure_2d=False)
        popularity = column_or_1d(check_array(X, ensure_2d=False, force_all_finite='allow-nan'))
        check_consistent_length(item_ids, popularity)

        self.popular_items = item_ids[(-popularity).argsort()]

        self.is_fitted_ = True
        return self

    def predict(self, X: numpy.ndarray, k: int) -> numpy.ndarray:
        check_is_fitted(self, 'is_fitted_')
        ids = column_or_1d(check_array(X, dtype=numpy.object_, ensure_2d=False))
        k = check_scalar(k,
                         name='output recommendations count',
                         target_type=int,
                         min_val=1,
                         max_val=len(self.popular_items))

        preds = []
        for item_id in range(len(ids)):
            preds.append(self.popular_items[self.popular_items != item_id][:k].tolist())

        return numpy.array(preds, dtype=numpy.object_)
