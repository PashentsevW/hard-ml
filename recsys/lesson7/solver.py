import argparse
import logging
from datetime import datetime
from typing import Optional

import boto3
import numpy
import pandas
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

import columns
import constants
from config import pipelines, score_wrapper, searchers
from utils.io.s3 import download_dataframe, upload_dataframe


def fit_and_eval(dataset_df: pandas.DataFrame,
                 pipeline_id: str,
                 sample_size: Optional[float] = None) -> Pipeline:
    pipeline = pipelines[pipeline_id]

    dataset_base_df = None
    if sample_size:
        all_users = dataset_df[columns.UID_COLUMN].unique()

        numpy.random.seed(constants.RANDOM_STATE)
        sample_users = numpy.random.choice(all_users, size=int(len(all_users) * sample_size), replace=False)

        dataset_base_df = dataset_df.loc[dataset_df[columns.UID_COLUMN].isin(sample_users), :]
    else:
        dataset_base_df = dataset_df.copy()

    logging.info('Got base data, with shape %s', dataset_base_df.shape)

    friends_count = dataset_base_df.groupby(columns.UID_COLUMN).size()
    target_users = friends_count[friends_count > 1].index

    dataset_flt_df = dataset_base_df.loc[dataset_base_df[columns.UID_COLUMN].isin(target_users), :]

    logging.info('Got filtered data, with shape %s', dataset_flt_df.shape)
        
    X = dataset_flt_df.loc[:, [columns.UID_COLUMN, columns.FRIEND_UID_COLUMN]].to_numpy()
    y = dataset_flt_df.loc[:, columns.UID_COLUMN].to_numpy()

    logging.info('Got X, y with shape: %s, %s', X.shape, y.shape)

    cv = StratifiedShuffleSplit(n_splits=1,
                                test_size=0.1,
                                random_state=constants.RANDOM_STATE)

    logging.info('Got cv: %s', cv)

    if pipeline_id in searchers:
        logging.info('Prepare searcher for "%s"', pipeline_id)

        searcher = searchers[pipeline_id][0](pipeline,
                                             cv=cv,
                                             **searchers[pipeline_id][1])
        search_result = searcher.fit(X, y)

        if search_result.best_params_:
            pipeline.set_params(**search_result.best_params_)

            logging.info('Search complete, find best params: %s', search_result.best_params_)
        else:
            raise ValueError('best_params_ not supported')
    else:
        logging.info('Got score: %s',
                     cross_val_score(pipeline, X, y, scoring=score_wrapper, cv=cv, verbose=4)[0])
    
    X_full = dataset_base_df.loc[:, [columns.UID_COLUMN, columns.FRIEND_UID_COLUMN]].to_numpy()

    logging.info('Got X_full with shape: %s', X_full.shape)
    
    return pipeline.fit(X_full)


def enrich_data_for_train(dataset_df: pandas.DataFrame) -> pandas.DataFrame:
    return pandas.concat([dataset_df,
                          dataset_df.rename(columns={columns.UID_COLUMN: columns.FRIEND_UID_COLUMN,
                                                     columns.FRIEND_UID_COLUMN: columns.UID_COLUMN})],
                         axis=0,
                         ignore_index=True)


def prepare_data_for_submit(pipeline: Pipeline,
                            dataset_df: pandas.DataFrame) -> pandas.DataFrame:
    users = dataset_df[columns.UID_COLUMN].unique()
    items = pipeline.predict(users, k=constants.AT_K)

    submission_df = pandas.DataFrame({
        columns.USER_ID_COLUMN: users
    })

    if items.ndim == 1:
        submission_df.loc[:, columns.Y_RECS_COLUMN] = (
            pandas.Series(items, index=submission_df.index)
            .apply(lambda row: numpy.array(row).flatten().tolist())
        )
    else:
        submission_df.loc[:, columns.Y_RECS_COLUMN] = (
            pandas.DataFrame(items, index=submission_df.index)
            .apply(lambda row: numpy.array(row).flatten().tolist(), axis=1)
        )

    return submission_df


if __name__ == '__main__':
    run_dt = datetime.now()
    
    argparser = argparse.ArgumentParser('lesson2')
    argparser.add_argument('-p', '--pipeline',
                           type=str,
                           choices=pipelines.keys(),
                           required=True,
                           dest='pipeline')
    argparser.add_argument('--sample-size',
                           default=None,
                           type=float,
                           required=False,
                           dest='sample_size')
    argparser.add_argument('--submit',
                           action='store_true',
                           dest='make_submit')
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

    dataset_df = download_dataframe(s3_client, constants.S3_BUCKET, constants.DATA_PATH / 'train.parquet')

    logging.info('Downloaded data, with shape %s', dataset_df.shape)

    dataset_df = enrich_data_for_train(dataset_df)
    
    logging.info('Got enriched data, with shape %s', dataset_df.shape)

    logging.info('Fit pipeline "%s"', args.pipeline)

    pipeline = fit_and_eval(dataset_df, args.pipeline, args.sample_size)

    logging.info('Got fitted pipeline:\n%s', pipeline)

    if args.make_submit:
        file_path = constants.SUBMISSION_PATH / args.pipeline / f'{run_dt.strftime("%Y%m%dT%H%M%S")}.parquet'

        submission_df = prepare_data_for_submit(pipeline, dataset_df)

        logging.info('Got submission with shape %s and columns %s', 
                    submission_df.shape,
                    submission_df.columns)

        upload_dataframe(submission_df, s3_client, constants.S3_BUCKET, file_path)

        logging.info('Recommendations saved to %s', file_path)

    logging.info('Done!')
