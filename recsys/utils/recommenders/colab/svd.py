import logging
from typing import Optional

import faiss
import numpy
import pandas
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (check_array,
                                      check_is_fitted,
                                      check_random_state,
                                      check_scalar,)
from surprise import AlgoBase, Dataset, Reader, SVD
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


class FunkSVDColabRecommender(BaseEstimator):
    def __init__(self,
                 n_factors: int = 100,
                 n_epochs: int = 20,
                 init_mean: float = 0,
                 init_std_dev: float = 0.1,
                 lr_all: float = 0.005,
                 reg_all: float = 0.02,
                 lr_bu: Optional[float] = None,
                 lr_bi: Optional[float] = None,
                 lr_pu: Optional[float] = None,
                 lr_qi: Optional[float] = None,
                 reg_bu: Optional[float] = None,
                 reg_bi: Optional[float] = None,
                 reg_pu: Optional[float] = None,
                 reg_qi: Optional[float] = None,
                 verbose: bool = False,
                 random_state: Optional[int] = None) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.lr_pu = lr_pu
        self.lr_qi = lr_qi
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: numpy.ndarray, y = None) -> 'FunkSVDColabRecommender':
        X = check_array(X)
        user_item_df = pandas.DataFrame(X, columns=['user_id', 'item_id'])
        user_item_df['rating'] = 5.

        model_params = {
            'n_factors': check_scalar(self.n_factors, name='n_factors', target_type=int, min_val=1),
            'n_epochs': check_scalar(self.n_epochs, name='n_epochs', target_type=int, min_val=1),
            'biased': False,
            'init_mean': check_scalar(self.init_mean, name='init_mean', target_type=(int, float), min_val=0),
            'init_std_dev': check_scalar(self.init_std_dev, name='init_std_dev', target_type=(int, float), min_val=0),
            'lr_all': check_scalar(self.lr_all, name='lr_all', target_type=float, min_val=1e-9),
            'reg_all': check_scalar(self.reg_all, name='reg_all', target_type=(int, float), min_val=1e-9),
            'lr_bu': (check_scalar(self.lr_bu, name='lr_bu', target_type=float, min_val=1e-9)
                      if self.lr_bu else None),
            'lr_bi': (check_scalar(self.lr_bi, name='lr_bi', target_type=float, min_val=1e-9)
                      if self.lr_bi else None),
            'lr_pu': (check_scalar(self.lr_pu, name='lr_pu', target_type=float, min_val=1e-9)
                      if self.lr_pu else None),
            'lr_qi': (check_scalar(self.lr_qi, name='lr_qi', target_type=float, min_val=1e-9)
                      if self.lr_qi else None),
            'reg_bu': (check_scalar(self.reg_bu, name='reg_bu', target_type=(int, float), min_val=1e-9)
                       if self.reg_bu else None),
            'reg_bi': (check_scalar(self.reg_bi, name='reg_bi', target_type=(int, float), min_val=1e-9)
                       if self.reg_bi else None),
            'reg_pu': (check_scalar(self.reg_pu, name='reg_pu', target_type=(int, float), min_val=1e-9)
                       if self.reg_pu else None),
            'reg_qi': (check_scalar(self.reg_qi, name='reg_qi', target_type=(int, float), min_val=1e-9)
                       if self.reg_qi else None),
            'verbose': check_scalar(self.verbose, name='verbose', target_type=bool),
            'random_state': check_random_state(self.random_state),
        }

        dataset = Dataset.load_from_df(user_item_df, Reader()).build_full_trainset()

        algo: AlgoBase = SVD(**model_params)
        algo.fit(dataset)

        self.dataset = dataset
        self.user_embeddings: numpy.ndarray = algo.pu
        self.item_embeddings: numpy.ndarray = algo.qi

        logging.info('Got user embeddings %s and item embeddings %s',
                     self.user_embeddings.shape,
                     self.item_embeddings.shape)

        if model_params.get('biased'):
            self.user_bias: numpy.ndarray = algo.bu
            self.item_bias: numpy.ndarray = algo.bi

            logging.info('Got user biases %s and item biases %s',
                         self.user_bias.shape,
                         self.item_bias.shape)

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

        X = check_array(X, dtype=None, ensure_2d=False)

        if X.ndim == 1:
            user_ids = numpy.unique(X)
        else:
            user_ids = numpy.unique(X[:, 0])
    
        k = check_scalar(k, name='output recommendations count', target_type=int, min_val=1)

        if progress_bar:
            logging.info('Get top%d items for %d users:', k, len(user_ids))
            
            user_ids = tqdm(user_ids)

        preds = []
        for user_id in user_ids:
            if user_id not in self.user_history:
                preds.append([])

            uid = self.dataset.to_inner_uid(user_id)

            _, y_rec = self.index.search(self.user_embeddings[uid, :].reshape(1, -1),
                                         k=k+len(self.user_history[user_id]))
            y_rec = y_rec[~numpy.isin(y_rec, self.user_history[user_id])][:k]

            y_rec = [self.dataset.to_raw_iid(yi) for yi in y_rec]

            preds.append(y_rec)

        return numpy.array(preds, dtype=numpy.object_)
