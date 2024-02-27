import logging

import numpy
import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import columns
import constants
from utils.recommenders.colab import PopularItemsColabRecommender
from utils.validation.metrics import ndcg_score

pipelines = {
    'baseline': Pipeline([('recommender', PopularItemsColabRecommender())]),
}


def score_wrapper(estimator: Pipeline, X: numpy.ndarray, y = None) -> float:
    if isinstance(estimator, Pipeline):
        y_true = (
            pandas.DataFrame(X, columns=[columns.USER_ID_COLUMN, columns.ARTIST_ID_COLUMN])
            .groupby(columns.USER_ID_COLUMN)[columns.ARTIST_ID_COLUMN]
            .agg(lambda items: items.to_list())
            .to_numpy(dtype=numpy.object_)
        )

        logging.info('Got y_true with shape %s', y_true.shape)

        y_pred = estimator.predict(X, k=constants.AT_K)

        logging.info('Got y_pred with shape %s', y_pred.shape)

        return ndcg_score(y_true, y_pred, constants.AT_K)
    else:
        raise ValueError(estimator)


searchers = {
    'experiment': (
        GridSearchCV,
        {
            'param_grid': {},
            'scoring': score_wrapper,
            'refit': False,
            'verbose': 4,
            'error_score': 'raise'
        }
    )
}
