import logging

import faiss
import numpy
import pandas
import torch
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (column_or_1d,
                                      check_array,
                                      check_is_fitted,
                                      check_scalar,)
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
from tqdm import tqdm


class Node2VecGraphRecommender(BaseEstimator):
    def __init__(self,
                 embedding_dim: int = 32,
                 walk_length: int = 10,
                 context_size: int = 5,
                 walks_per_node: int = 5,
                 num_negative_samples: int = 1,
                 p: float = 1.0,
                 q: float = 1.0,
                 sparse: bool = True,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 lr: float = 0.01,
                 n_epochs: int = 5,
                 num_workers: int = 1,
                 device: str = 'cpu',
                 random_state: int = None) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size 
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.p = p
        self.q = q
        self.sparse = sparse
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lr = lr
        self.n_epochs = n_epochs
        self.num_workers = num_workers
        self.device = device
        self.random_state = random_state

    def fit(self, X: numpy.ndarray, y = None) -> 'Node2VecGraphRecommender':
        X = check_array(X, dtype=None)

        logging.info('Fit on %s', self.device.upper())

        data = Data(edge_index=torch.from_numpy(X).T.contiguous())
        data.validate()
        data.to(self.device)

        if self.walk_length < self.context_size:
            logging.warning('walk_length < context_size, set context_size = walk_length')

            self.context_size = self.walk_length

        model = Node2Vec(
            data.edge_index,
            embedding_dim=check_scalar(self.embedding_dim, name='embedding_dim', target_type=int, min_val=1),
            walk_length=check_scalar(self.walk_length, name='walk_length', target_type=int, min_val=1),
            context_size=check_scalar(self.context_size, name='context_size', target_type=int, min_val=1),
            walks_per_node=check_scalar(self.walks_per_node, name='walks_per_node', target_type=int, min_val=1),
            num_negative_samples=check_scalar(self.num_negative_samples, name='num_negative_samples', target_type=int, min_val=1),
            p=check_scalar(self.p, name='p', target_type=float, min_val=0),
            q=check_scalar(self.q, name='q', target_type=float, min_val=0),
            sparse=check_scalar(self.sparse, name='sparse', target_type=bool),
        )
        model.to(self.device)

        torch.manual_seed(self.random_state)

        loader = model.loader(
            batch_size=check_scalar(self.batch_size, name='batch_size', target_type=int, min_val=1),
            shuffle=check_scalar(self.shuffle, name='shuffle', target_type=bool),
            num_workers=check_scalar(self.num_workers, name='num_workers', target_type=int, min_val=1)
        )
        optimizer = torch.optim.SparseAdam(
            list(model.parameters()),
            lr=check_scalar(self.lr, name='lr', target_type=float, min_val=1e-5)
        )
        
        for epoch in range(check_scalar(self.n_epochs, 'n_epochs', target_type=int, min_val=1)):
            model.train()

            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()

                loss = model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                
                optimizer.step()
                
                total_loss += loss.item()
            
            logging.info('Epoch #%d, loss: %f', epoch, total_loss / len(loader))

        self.embeddings = model.embedding.weight.detach().cpu().numpy()

        logging.info('Got embeddings with shape %s', self.embeddings.shape)

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        self.user_history = (pandas.DataFrame(X, columns=['user_id', 'item_id'])
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
                continue
            
            _, y_rec = self.index.search(self.embeddings[user_id, :].reshape(1, -1),
                                         k=k+len(self.user_history[user_id]))
            y_rec = y_rec[~numpy.isin(y_rec, self.user_history[user_id])][:k]

            preds.append(y_rec)

        return numpy.array(preds, dtype=numpy.object_)
