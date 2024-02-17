import argparse
import logging
from datetime import datetime

import boto3
import numpy
import pandas
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline

from utils.io.s3 import download_dataframe, upload_dataframe
from utils.text.models import pretrained_models
from utils.text.models.loader import GensimModelLoader
from utils.validation.metrics import hitrate_score

import columns
import constants
from config import pipelines, searchers

def fit_pipeline(train_df: pandas.DataFrame,
                 valid_df: pandas.DataFrame,
                 pipeline_id: str) -> Pipeline:
    pipeline = pipelines[pipeline_id]

    if pipeline_id in searchers:
        logging.info('Prepare searcher for "%s"', pipeline_id)
        
        X = pandas.concat([train_df,
                           valid_df[[columns.TITLE_ID_COLUMN]].merge(train_df, on=columns.TITLE_ID_COLUMN)],
                          ignore_index=True)
        
        logging.info('Got X, with shape %s', X.shape)

        y = pandas.concat([train_df.loc[:, columns.TITLE_ID_COLUMN],
                           valid_df.loc[:, columns.RELEVANT_TITLES_COLUMN].apply(lambda titles: titles.tolist())],
                          ignore_index=True)
        y = y.to_numpy(dtype=numpy.object_)

        logging.info('Got y, with shape %s', y.shape)

        test_fold = numpy.hstack([numpy.zeros(train_df.shape[0], dtype=numpy.int_),
                                  numpy.ones(valid_df.shape[0], dtype=numpy.int_)])
        test_fold -= 1

        searcher = searchers[pipeline_id][0](pipeline,
                                             cv=PredefinedSplit(test_fold),
                                             **searchers[pipeline_id][1])
        search_result = searcher.fit(X, y)

        if search_result.best_params_:
            pipeline.set_params(**search_result.best_params_)

            logging.info('Search complete, find best params: %s', search_result.best_params_)
        else:
            raise ValueError('best_params_ not supported')
    
    return pipeline.fit(train_df, train_df[columns.TITLE_ID_COLUMN].to_numpy())


if __name__ == '__main__':
    run_dt = datetime.now()
    
    argparser = argparse.ArgumentParser('lesson2')
    argparser.add_argument('-p', '--pipeline',
                           type=str,
                           choices=pipelines.keys(),
                           required=True,
                           dest='pipeline')
    argparser.add_argument('-v', '--verbose',
                           default='INFO',
                           type=lambda arg: logging.getLevelName(arg),
                           required=False,
                           dest='loglevel')

    args = argparser.parse_args()

    logging.basicConfig(level=args.loglevel)
    logging.debug(args.__dict__)

    logging.info('Use "%s" pipeline', args.pipeline)

    s3_session = boto3.session.Session()
    s3_client = s3_session.client(service_name='s3', endpoint_url='https://storage.yandexcloud.net')

    logging.info('Download word models')

    pretrained_models.register('glove-wiki-gigaword-50', GensimModelLoader(), 50)

    logging.info('Downloaded %d models', len(pretrained_models))

    logging.info('Download data')

    train_df = download_dataframe(s3_client, constants.S3_BUCKET, constants.DATA_PATH / 'imdb_movies.parquet')

    logging.info('Downloaded train data, with shape %s', train_df.shape)
    
    test_df = download_dataframe(s3_client, constants.S3_BUCKET, constants.DATA_PATH / 'relevant_titles_subsample.parquet')

    logging.info('Downloaded test data, with shape %s', test_df.shape)

    logging.info('Fit pipeline "%s"', args.pipeline)

    pipeline = fit_pipeline(train_df, test_df, args.pipeline)

    logging.info('Pipeline fitted,\n%s', pipeline)

    logging.info('Eval pipeline "%s"', args.pipeline)

    y_true = (test_df[columns.RELEVANT_TITLES_COLUMN]
              .apply(lambda items: items.tolist())
              .to_numpy(dtype=numpy.object_))
    y_pred = pipeline[-1].predict(test_df[columns.TITLE_ID_COLUMN].to_numpy().reshape(-1, 1), k=constants.AT_K)

    logging.info('Metric: %f', hitrate_score(y_true, y_pred, constants.AT_K))

    logging.info('Save recommendations')

    y_pred = pipeline[-1].predict(train_df[columns.TITLE_ID_COLUMN].to_numpy().reshape(-1, 1), k=constants.AT_K)

    submission_df = train_df.loc[:, [columns.TITLE_ID_COLUMN]]

    if y_pred.ndim == 1:
        submission_df.loc[:, columns.RECS_COLUMN] = (
            pandas.Series(y_pred, index=submission_df.index)
            .apply(lambda row: numpy.array(row).flatten().tolist())
        )
    else:
        submission_df.loc[:, columns.RECS_COLUMN] = (
            pandas.DataFrame(y_pred, index=submission_df.index)
            .apply(lambda row: numpy.array(row).flatten().tolist(), axis=1)
        )

    logging.info('Got submissions data, with shape %s', submission_df.shape)

    file_path = constants.SUBMISSION_PATH / args.pipeline / f'{run_dt.strftime("%Y%m%dT%H%M%S")}.parquet'

    upload_dataframe(submission_df, s3_client, constants.S3_BUCKET, file_path)

    logging.info('Recommendations saved to %s', file_path)
    logging.info('Done!')
