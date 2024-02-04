import numpy
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_scalar, column_or_1d


class RandomItemsContentRecommender(BaseEstimator):
    def __init__(self, random_state: int) -> None:
        super().__init__()
        self.random_state = random_state

    def fit(self, X: numpy.ndarray, y: numpy.ndarray) -> 'RandomItemsContentRecommender':
        self.item_ids = check_array(y, dtype=None, ensure_2d=False)
        self.rng = numpy.random.default_rng(self.random_state)
        self.is_fitted_ = True
        return self

    def predict(self, X: numpy.ndarray, k: int) -> numpy.ndarray:
        check_is_fitted(self, 'is_fitted_')
        ids = column_or_1d(check_array(X, dtype=numpy.object_, ensure_2d=False))
        k = check_scalar(k,
                         name='output recommendations count',
                         target_type=int,
                         min_val=1,
                         max_val=len(self.item_ids))

        preds = []
        for i in range(ids.shape[0]):
            if ids[i] not in self.item_ids:
                preds.append([])
                continue

            preds.append(self.rng.choice(self.item_ids, size=k).tolist())
        
        return numpy.array(preds, dtype=numpy.object_)
