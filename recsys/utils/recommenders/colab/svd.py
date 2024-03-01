import logging

import numpy
import pandas
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (check_array,
                                      check_is_fitted,
                                      check_random_state,
                                      check_scalar,)
from tqdm import tqdm


def _user_item_to_sparse(user_item_df: pandas.DataFrame) -> csr_matrix:
    rows = user_item_df['user_id'].to_numpy(dtype=numpy.int_)
    cols = user_item_df['item_id'].to_numpy(dtype=numpy.int_)
    data = numpy.ones(rows.shape, dtype=numpy.float_)

    return coo_matrix((data, (rows, cols))).tocsr()


class PureSVDColabRecommender(BaseEstimator):
    def __init__(self,
                 n_components: int,
                 random_state: int) -> None:
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: numpy.ndarray, y = None) -> 'PureSVDColabRecommender':
        X = check_array(X)
        user_item_df = pandas.DataFrame(X, columns=['user_id', 'item_id'])

        n_components = check_scalar(self.n_components,
                                    name='number of components',
                                    target_type=int,
                                    min_val=1)
        random_state = check_random_state(self.random_state)

        U, S, V = svds(_user_item_to_sparse(user_item_df),
                       k=n_components,
                       random_state=random_state)
        self.user_embeddings: numpy.ndarray = U @ numpy.diag(S)
        self.item_embeddings: numpy.ndarray = V.T

        logging.info('Got user embeddings %s and item embeddings %s',
                     self.user_embeddings.shape,
                     self.item_embeddings.shape)
        
        self.user_history = (user_item_df
                             .groupby('user_id')['item_id']
                             .agg(lambda items: items.to_list())
                             .to_dict())

        self.is_fitted_ = True
        return self

    def predict(self, X: numpy.ndarray, k: int) -> numpy.ndarray:
        check_is_fitted(self, 'is_fitted_')

        X = check_array(X, dtype=None, ensure_2d=False)

        if X.ndim == 1:
            user_ids = numpy.unique(X)
        else:
            user_ids = numpy.unique(X[:, 0])
    
        k = check_scalar(k,
                         name='output recommendations count',
                         target_type=int,
                         min_val=1,
                         max_val=self.item_embeddings.shape[0])

        logging.info('Get top%d items for %d users:', k, len(user_ids))

        preds = []
        for user_id in tqdm(user_ids):
            if user_id not in self.user_history:
                preds.append([])

            similarities = self.user_embeddings[user_id, :] @ self.item_embeddings.T

            y_rec = similarities.argsort()[::-1][:k + len(self.user_history[user_id])]
            y_rec = y_rec[~numpy.isin(y_rec, self.user_history[user_id])][:k]

            preds.append(y_rec)

        return numpy.array(preds, dtype=numpy.object_)
