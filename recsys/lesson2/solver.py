import argparse
import pathlib
import logging
from datetime import datetime

import boto3
import numpy
import pandas
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from utils.io.s3 import download_dataframe, upload_dataframe
from utils.recommenders.content import (PopularItemsContentRecommender,
                                        RandomItemsContentRecommender,
                                        SimilarItemsContentRecommender)
from utils.validation.metrics import hitrate_score

s3_bucket = 'cloud-pashentsevw-default'
s3_folder = pathlib.Path('hardml/recsys/lesson2/')

random_state = 94

data_path = s3_folder
submission_path = s3_folder / 'submissions'

title_id_column = 'title_id'
relevant_titles_column = 'relevant_titles'

at_k = 10

pipelines = {
    'random': Pipeline([('recommender', RandomItemsContentRecommender(random_state))]),
    'popular': Pipeline([('popularity_transformer',
                           make_column_transformer((SimpleImputer(strategy='constant', fill_value=-1),
                                                    ['rating_value']),
                                                   remainder='drop')),
                          ('recommender', PopularItemsContentRecommender())]),
    'keywords': Pipeline([
        ('keywords_vectorize',
         make_column_transformer((CountVectorizer(lowercase=False,
                                                  tokenizer=lambda keywords: keywords,
                                                  token_pattern=None),
                                  'keywords'),
                                 remainder='drop')),
        ('normalizer', Normalizer()),
        ('recommender', SimilarItemsContentRecommender())]),
    'similarity': Pipeline([
        ('keywords_vectorize',
         make_column_transformer((CountVectorizer(lowercase=False,
                                                  tokenizer=lambda keywords: keywords,
                                                  token_pattern=None),
                                  'keywords'),
                                 remainder='drop')),
        ('normalizer', Normalizer()),
        ('recommender', SimilarItemsContentRecommender(distance_metric='cosine'))])
}


def score_wrapper(estimator, X, y) -> float:
    if isinstance(estimator, Pipeline):
        y_pred = estimator[-1].predict(X[title_id_column].to_numpy().reshape(-1, 1), k=at_k)
        return hitrate_score(y, y_pred, at_k)
    else:
        raise ValueError(estimator)


searchers = {
    'similarity': (
        GridSearchCV,
        {'param_grid': { 'recommender__distance_metric': ['cosine', 'l1', 'l2', 'euclidean']},
         'scoring': score_wrapper,
         'refit': False,
         'verbose': 4})
}


def fit_pipeline(train_df: pandas.DataFrame,
                 valid_df: pandas.DataFrame,
                 pipeline_id: str) -> Pipeline:
    pipeline = pipelines[pipeline_id]

    if pipeline_id in searchers:
        logging.info('Prepare searcher for "%s"', pipeline_id)
        
        X = pandas.concat([train_df,
                           valid_df[[title_id_column]].merge(train_df, on=title_id_column)],
                          ignore_index=True)
        
        logging.info('Got X, with shape %s', X.shape)

        y = pandas.concat([train_df.loc[:, title_id_column],
                           valid_df.loc[:, relevant_titles_column].apply(lambda titles: titles.tolist())],
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
            raise ValueError('best_estimator_ not supported')
    
    return pipeline.fit(train_df, train_df[title_id_column].to_numpy())

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

    logging.info('Download data')

    train_df = download_dataframe(s3_client, s3_bucket, data_path / 'imdb_movies.parquet')

    logging.info('Downloaded train data, with shape %s', train_df.shape)
    
    test_df = download_dataframe(s3_client, s3_bucket, data_path / 'relevant_titles_subsample.parquet')

    logging.info('Downloaded test data, with shape %s', test_df.shape)

    logging.info('Fit pipeline "%s"', args.pipeline)

    pipeline = fit_pipeline(train_df, test_df, args.pipeline)

    logging.info('Pipeline fitted,\n%s', pipeline)

    logging.info('Eval pipeline "%s"', args.pipeline)

    y_true = (test_df[relevant_titles_column]
              .apply(lambda items: items.tolist())
              .to_numpy(dtype=numpy.object_))
    y_pred = pipeline[-1].predict(test_df[title_id_column].to_numpy().reshape(-1, 1), k=at_k)

    logging.info('Metric: %f', hitrate_score(y_true, y_pred, at_k))

    logging.info('Save recommendations')

    submission_df = test_df
    submission_df.loc[:, relevant_titles_column] = (
        pandas.DataFrame(y_pred).apply(lambda row: row.tolist(), axis=1)
    )

    logging.info('Got submissions data, with shape %s', submission_df.shape)

    file_path = submission_path / args.pipeline / f'{run_dt.strftime("%Y%m%dT%H%M%S")}.parquet'

    upload_dataframe(submission_df, s3_client, s3_bucket, file_path)

    logging.info('Recommendations saved to %s', file_path)
    logging.info('Done!')
