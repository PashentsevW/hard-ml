import logging

import boto3
import numpy
import pandas
from sklearn.pipeline import Pipeline

from utils.features.calculation.warehouse import S3DataWarehouse
from utils.validation.metrics import ndcg_score, recall_score

import calcers
import columns
import constants
import estimators

s3_session = boto3.session.Session()
s3_client = s3_session.client(service_name='s3', endpoint_url='https://storage.yandexcloud.net')

dwh = S3DataWarehouse(s3_client)

splitter = {
    'by_sessions': estimators.LastNSampleSplitter(n_last=2),
}

cand_pipelines = {
    'als': ...,
    'w2v': ...,
}

cand_data_pipelines = {
    'user_item': calcers.UserItemMatrixCalcer('user_item_matrix', dwh),
}

rank_pipelines = {
    'popularity': ...,
    'catboost': ...,
}
rank_pipelines['baseline'] = rank_pipelines['popularity']

rank_data_pipelines = {
    'popularity_by_users': ...,
}
rank_data_pipelines['baseline'] = rank_data_pipelines['popularity_by_users']


def cand_score_wrapper(estimator: Pipeline,
                       X: numpy.ndarray,
                       y: numpy.ndarray = None,
                       group: numpy.ndarray = None,) -> float:
    if isinstance(estimator, Pipeline):
        logging.info('Evaluate')

        user_items = (
            pandas.DataFrame(X, columns=[columns.USER_ID_COLUMN, columns.ITEM_ID_COLUMN])
            .groupby(columns.USER_ID_COLUMN)[columns.ITEM_ID_COLUMN]
            .agg(lambda items: items.to_list())
        )

        y_true = user_items.to_numpy(dtype=numpy.object_)

        logging.info('Got y_true with shape %s', y_true.shape)

        y_pred = estimator.predict(user_items.index.to_numpy(), k=constants.AT_K_CAND)

        logging.info('Got y_pred with shape %s', y_pred.shape)

        return recall_score(y_true, y_pred, constants.AT_K_CAND)
    else:
        raise ValueError(estimator)


def rank_score_wrapper(estimator: Pipeline,
                       X: numpy.ndarray,
                       y: numpy.ndarray,
                       group: numpy.ndarray) -> float:
    if isinstance(estimator, Pipeline):
        logging.info('Evaluate')

        relevances = estimator.predict(X)

        logging.info('Got relevances with shape %s', relevances.shape)

        eval_df = pandas.DataFrame({columns.GROUP_COLUMN: group,
                                    columns.Y_TRUE_COLUMN: y,
                                    columns.Y_PRED_COLUMN: relevances,})

        # TODO change y from relevance to item_ids
        
        y_true = (eval_df
                  .groupby(columns.GROUP_COLUMN)[columns.Y_TRUE_COLUMN]
                  .agg(lambda scores: scores.to_list())
                  .to_numpy(dtype=numpy.object_))

        logging.info('Got y_true with shape %s', y_true.shape)
        
        y_pred = (eval_df
                  .groupby(columns.GROUP_COLUMN)[columns.Y_PRED_COLUMN]
                  .agg(lambda scores: scores.to_list())
                  .to_numpy(dtype=numpy.object_))

        logging.info('Got y_pred with shape %s', y_pred.shape)

        return ndcg_score(y_true, y_pred, constants.AT_K_RANK)
    else:
        raise ValueError(estimator)


searchers = {}
