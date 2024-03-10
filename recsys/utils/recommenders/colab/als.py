import logging
from typing import Optional

import faiss
import numpy
import pandas
from lightfm import LightFM
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (column_or_1d,
                                      check_array,
                                      check_is_fitted,
                                      check_random_state,
                                      check_scalar,)
from tqdm import tqdm


def _user_item_to_sparse(user_item_df: pandas.DataFrame) -> coo_matrix:
    rows = user_item_df['user_id'].to_numpy(dtype=numpy.int_)
    cols = user_item_df['item_id'].to_numpy(dtype=numpy.int_)
    data = numpy.ones(rows.shape, dtype=numpy.float_)

    return coo_matrix((data, (rows, cols)))


class LightFMColabRecommender(BaseEstimator):
    def __init__(self,
                 no_components: int = 10,
                 k: int = 5,
                 n: int = 10,
                 learning_schedule: str = 'adagrad',
                 loss: str = 'logistic',
                 learning_rate: float = 0.05,
                 rho: float = 0.95,
                 epsilon: float = 1e-6,
                 item_alpha: float = 0.0,
                 user_alpha: float = 0.0,
                 max_sampled: int = 10,
                 epochs: int = 1,
                 num_threads: int = 1,
                 verbose: bool = False,
                 random_state: Optional[int] = None) -> None:
        super().__init__()
        self.no_components = no_components
        self.k = k
        self.n = n
        self.learning_schedule = learning_schedule
        self.loss = loss
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.max_sampled = max_sampled
        self.epochs = epochs
        self.num_threads = num_threads
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: numpy.ndarray, y = None) -> 'LightFMColabRecommender':
        X = check_array(X)
        user_item_df = pandas.DataFrame(X, columns=['user_id', 'item_id'])

        model_params = {
            'no_components': check_scalar(self.no_components, name='no_components', target_type=int, min_val=1),
            'k': check_scalar(self.k, name='k', target_type=int, min_val=1),
            'n': check_scalar(self.n, name='n', target_type=int, min_val=1),
            'learning_schedule': check_scalar(self.learning_schedule, name='learning_schedule', target_type=str),
            'loss': check_scalar(self.loss, name='loss', target_type=str),
            'learning_rate': check_scalar(self.learning_rate, name='learning_rate', target_type=float, min_val=1e-9),
            'rho': check_scalar(self.rho, name='rho', target_type=float),
            'epsilon': check_scalar(self.epsilon, name='epsilon', target_type=float),
            'item_alpha': check_scalar(self.item_alpha, name='item_alpha', target_type=(float, int)),
            'user_alpha': check_scalar(self.user_alpha, name='user_alpha', target_type=(float, int)),
            'max_sampled': check_scalar(self.max_sampled, name='max_sampled', target_type=int),
            'random_state': check_random_state(self.random_state),
        }
        fit_params = {
            'epochs': check_scalar(self.epochs, name='epochs', target_type=int, min_val=1),
            'num_threads': check_scalar(self.num_threads, name='num_threads', target_type=int, min_val=-1),
            'verbose': check_scalar(self.verbose, name='verbose', target_type=bool),
        }

        model = LightFM(**model_params)
        model.fit(interactions=_user_item_to_sparse(user_item_df), **fit_params)
        
        self.user_embeddings: numpy.ndarray = model.user_embeddings
        self.item_embeddings: numpy.ndarray = model.item_embeddings

        logging.info('Got user embeddings %s and item embeddings %s',
                     self.user_embeddings.shape,
                     self.item_embeddings.shape)

        self.index = faiss.IndexFlatIP(self.item_embeddings.shape[1])
        self.index.add(self.item_embeddings)

        logging.info('Builded search index')
        
        self.user_history = (user_item_df
                             .groupby('user_id')['item_id']
                             .agg(lambda items: items.to_list())
                             .to_dict())

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

            _, y_rec = self.index.search(self.user_embeddings[user_id, :].reshape(1, -1),
                                         k=k+len(self.user_history[user_id]))
            y_rec = y_rec[~numpy.isin(y_rec, self.user_history[user_id])][:k]

            preds.append(y_rec)

        return numpy.array(preds, dtype=numpy.object_)
