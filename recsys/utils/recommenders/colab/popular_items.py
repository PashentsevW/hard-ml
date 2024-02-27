import numpy
import pandas
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (check_array,
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
                              .index.to_numpy())
        
        self.user_history = (user_item_df
                             .groupby('user_id')['item_id']
                             .agg(lambda items: items.to_list())
                             .to_dict())

        self.is_fitted_ = True
        return self

    def predict(self, X: numpy.ndarray, k: int) -> numpy.ndarray:
        check_is_fitted(self, 'is_fitted_')

        X = check_array(X, dtype=None)
        user_item_df = pandas.DataFrame(X, columns=['user_id', 'item_id'])
    
        k = check_scalar(k,
                         name='output recommendations count',
                         target_type=int,
                         min_val=1,
                         max_val=len(self.popular_items))

        preds = []
        for user_id in user_item_df.loc[:, 'user_id']:
            if user_id not in self.user_history:
                continue

            preds.append(self.popular_items[~numpy.isin(self.user_history[user_id], self.popular_items)][:k])

        return numpy.array(preds, dtype=numpy.object_)

