import logging

import numpy
import pandas
import torch
from optuna.distributions import CategoricalDistribution, IntDistribution, FloatDistribution
from optuna.integration import OptunaSearchCV
from sklearn.pipeline import Pipeline

import columns
import constants
from utils.recommenders.graph import (Node2VecGraphRecommender,
                                      RandomItemsGraphRecommender)
from utils.validation.metrics import recall_score

pipelines = {
    'random': Pipeline([('recommender', RandomItemsGraphRecommender(random_state=constants.RANDOM_STATE))]),
    'node2vec': Pipeline(
        [('recommender', 
          Node2VecGraphRecommender(device='cuda' if torch.cuda.is_available() else 'cpu',
                                   random_state=constants.RANDOM_STATE))]
    )
}


def score_wrapper(estimator: Pipeline, X: numpy.ndarray, y: numpy.ndarray = None) -> float:
    if isinstance(estimator, Pipeline):
        logging.info('Evaluate')

        user_items = (
            pandas.DataFrame(X, columns=[columns.UID_COLUMN, columns.FRIEND_UID_COLUMN])
            .groupby(columns.UID_COLUMN)[columns.FRIEND_UID_COLUMN]
            .agg(lambda items: items.to_list())
        )

        y_true = user_items.to_numpy(dtype=numpy.object_)

        logging.info('Got y_true with shape %s', y_true.shape)

        y_pred = estimator.predict(user_items.index.to_numpy(), k=constants.AT_K)

        logging.info('Got y_pred with shape %s', y_pred.shape)

        return recall_score(y_true, y_pred, constants.AT_K)
    else:
        raise ValueError(estimator)


searchers = {
    'node2vec': (
        OptunaSearchCV,
        {
            'param_distributions': {
                'recommender__embedding_dim': CategoricalDistribution(choices=[16, 32, 64, 128]),
                'recommender__walk_length': IntDistribution(low=2, high=6),
                'recommender__walks_per_node': IntDistribution(low=1, high=20),
                # 'recommender__context_size': IntDistribution(low=1, high=20),
                'recommender__num_negative_samples': IntDistribution(low=5, high=20),
                'recommender__p': FloatDistribution(low=0.01, high=1),
                'recommender__q': FloatDistribution(low=0.01, high=1),
            },
            'n_jobs': 6,
            'n_trials': 50,
            'scoring': score_wrapper,
            'refit': False,
            'verbose': 4,
            'random_state': constants.RANDOM_STATE,
            'error_score': 'raise'
        }
    ),
}
