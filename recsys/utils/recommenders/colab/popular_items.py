import numpy
import pandas
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (check_array,
                                      check_consistent_length,
                                      check_is_fitted,
                                      check_scalar,
                                      column_or_1d)


class PopularItemsColabRecommender(BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: numpy.ndarray, y: numpy.ndarray) -> 'PopularItemsColabRecommender':
        user_ids = column_or_1d(check_array(X, ensure_2d=False))
        item_ids = column_or_1d(check_array(y, dtype=None, ensure_2d=False))
        check_consistent_length(user_ids, item_ids)

        user_item_df = pandas.DataFrame({'user_id': user_ids, 'item_id': item_ids}).explode('item_id')

        self.popular_items = (user_item_df
                              .groupby('item_id')['user_id'].count()
                              .sort_values(ascending=False)
                              .index.to_numpy())
        
        self.user_items = (user_item_df
                           .groupby('user_id')['item_id']
                           .agg(lambda items: items.to_list())
                           .to_dict())

        self.is_fitted_ = True
        return self

    def predict(self, X: numpy.ndarray, k: int) -> numpy.ndarray:
        check_is_fitted(self, 'is_fitted_')
        user_ids = column_or_1d(check_array(X, dtype=numpy.object_, ensure_2d=False))
        k = check_scalar(k,
                         name='output recommendations count',
                         target_type=int,
                         min_val=1,
                         max_val=len(self.popular_items))

        preds = []
        for user_id in range(len(user_ids)):
            if user_id not in self.user_items:
                continue

            preds.append(self.popular_items[~numpy.isin(self.user_items[user_id], self.popular_items)][:k])

        return numpy.array(preds, dtype=numpy.object_)

