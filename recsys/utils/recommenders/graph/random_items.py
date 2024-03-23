import logging

import numpy
import pandas
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (column_or_1d,
                                      check_array,
                                      check_is_fitted,
                                      check_scalar,)
from tqdm import tqdm


class RandomItemsGraphRecommender(BaseEstimator):
    def __init__(self, random_state: int) -> None:
        super().__init__()
        self.random_state = random_state 

    def fit(self, X: numpy.ndarray, y = None) -> 'RandomItemsGraphRecommender':
        X = check_array(X, dtype=None)
        self.rng = numpy.random.default_rng(self.random_state)

        user_item_df = pandas.DataFrame(X, columns=['user_id', 'item_id'])
                
        user_history = (user_item_df
                        .groupby('user_id')['item_id']
                        .agg(lambda items: items.to_list()))
        
        user_history_lenght = user_history.map(len).to_numpy(dtype=numpy.float_)
        user_recs = self.rng.choice(user_history.index,
                                    size=(user_history.shape[0], int(user_history_lenght.max())),
                                    p=user_history_lenght / user_history_lenght.sum())
                
        user_recs = pandas.DataFrame(user_recs, index=user_history.index)
        self.user_recs = user_recs.apply(lambda items: items.to_list(), axis=1).to_dict()

        self.user_history = user_history.to_dict()

        self.is_fitted_ = True
        return self

    def predict(self, X: numpy.ndarray, k: int, progress_bar: bool = True) -> numpy.ndarray:
        check_is_fitted(self, 'is_fitted_')

        user_ids = column_or_1d(check_array(X, dtype=None, ensure_2d=False))    
        k = check_scalar(k, name='k', target_type=int, min_val=1)

        if progress_bar:
            logging.info('Get top%d items for %d users:', k, len(user_ids))

            user_ids = tqdm(user_ids)

        preds = []
        for user_id in user_ids:
            if user_id not in self.user_history:
                preds.append([])
                continue
            
            y_rec = numpy.array(self.user_recs[user_id])
            y_rec = y_rec[~numpy.isin(y_rec, self.user_history[user_id])][:k]

            preds.append(y_rec)

        return numpy.array(preds, dtype=numpy.object_)
