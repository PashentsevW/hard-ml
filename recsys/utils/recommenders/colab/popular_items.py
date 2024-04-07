import numpy
import pandas
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (column_or_1d,
                                      check_array,
                                      check_is_fitted,
                                      check_scalar,)


class PopularItemsColabRecommender(BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: numpy.ndarray, y = None) -> 'PopularItemsColabRecommender':
        X = check_array(X, dtype=None)
        user_item_df = pandas.DataFrame(X, columns=['user_id', 'item_id'])

        self.popular_items = (user_item_df
                              .groupby('item_id')['user_id'].count()
                              .sort_values(ascending=False)
                              .index
                              .to_numpy())
        
        self.user_history = (user_item_df
                             .groupby('user_id')['item_id']
                             .agg(lambda items: items.to_list())
                             .to_dict())

        self.is_fitted_ = True
        return self

    def predict(self, X: numpy.ndarray, k: int) -> numpy.ndarray:
        check_is_fitted(self, 'is_fitted_')

        user_ids = column_or_1d(check_array(X, dtype=None, ensure_2d=False))    
        k = check_scalar(k, name='k', target_type=int, min_val=1)

        preds = []
        for user_id in user_ids:
            if user_id not in self.user_history:
                preds.append([])

            y_rec = self.popular_items[:k + len(self.user_history[user_id])]
            y_rec = y_rec[~numpy.isin(y_rec, self.user_history[user_id])][:k]

            preds.append(y_rec)

        return numpy.array(preds, dtype=numpy.object_)
