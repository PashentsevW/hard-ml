import logging

import numpy
import pandas
from sklearn.pipeline import Pipeline

import columns
import constants
from utils.recommenders.graph import RandomItemsGraphRecommender
from utils.validation.metrics import recall_score

pipelines = {
    'random': Pipeline([('recommender', RandomItemsGraphRecommender(random_state=constants.RANDOM_STATE))])
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


searchers = {}
