import logging

import numpy
import pandas
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (column_or_1d,
                                      check_array,
                                      check_is_fitted,
                                      check_scalar,)
from tqdm import tqdm


class Word2VecColabRecommender(BaseEstimator):
    def __init__(self,
                 sg: int = 0,
                 window: int = 5,
                 ns_exponent: float = 0.75,
                 negative: int = 5,
                 min_count: int = 5,
                 vector_size: int = 100,
                 epochs: int = 5,
                 workers: int = 3,
                 seed: int = 1) -> None:
        super().__init__()
        self.sg = sg
        self.window = window
        self.ns_exponent = ns_exponent
        self.negative = negative
        self.min_count = min_count
        self.vector_size = vector_size
        self.epochs = epochs
        self.workers = workers
        self.seed = seed

    def fit(self, X: numpy.ndarray, y = None) -> 'Word2VecColabRecommender':
        X = check_array(X)
        user_item_df = pandas.DataFrame(X, columns=['user_id', 'item_id'])

        self.user_history = (user_item_df
                             .groupby('user_id')['item_id']
                             .agg(lambda items: items.to_list())
                             .to_dict())

        model_params = {
            'sg': check_scalar(self.sg, name='sg', target_type=int),
            'window': check_scalar(self.window, name='window', target_type=int),
            'ns_exponent': check_scalar(self.ns_exponent, name='ns_exponent', target_type=(float, int)),
            'negative': check_scalar(self.negative, name='negative', target_type=int),
            'min_count': check_scalar(self.min_count, name='min_count', target_type=int),
            'vector_size': check_scalar(self.vector_size, name='vector_size', target_type=int),
            'epochs': check_scalar(self.epochs, name='epochs', target_type=int),
            'workers': check_scalar(self.workers, name='workers', target_type=int),
            'seed': check_scalar(self.seed, name='seed', target_type=int),
        }

        self.model = Word2Vec(sentences=list(self.user_history.values()),
                              **model_params)

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

            y_rec = self.model.predict_output_word(
                self.user_history[user_id][-self.model.window:],
                k + len(self.user_history[user_id])
            )

            if y_rec is None:
                preds.append([])
                continue

            y_rec = [item_id
                     for item_id, _ in y_rec
                     if item_id not in self.user_history[user_id]][:k]

            preds.append(y_rec)

        return numpy.array(preds, dtype=numpy.object_)
