import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Sequence, Tuple

import boto3
import numpy
import pandas
from sklearn.model_selection import PredefinedSplit, cross_val_score
from sklearn.pipeline import Pipeline

import columns
import constants
from config import pipelines, searchers
from utils.io.s3 import download_dataframe, upload_dataframe


@dataclass
class Mapping:
    id2code: Dict[str, int]
    code2id: Dict[int, str]

    @staticmethod
    def generate_mapping(source_ids: Sequence[str]) -> 'Mapping':
        id2code = {}
        code2id = {}
        for ind, id_ in enumerate(source_ids):
            id2code[id_] = ind
            code2id[ind] = id_
        
        return Mapping(id2code, code2id)


def preprocess_dataset(
        dataset_df: pandas.DataFrame
    ) -> Tuple[pandas.DataFrame, Mapping, Mapping]:
    users_mapping = Mapping.generate_mapping(dataset_df[columns.USER_ID_COLUMN].to_numpy())
    artists_mapping = Mapping.generate_mapping(dataset_df[columns.ARTIST_ID_COLUMN].to_numpy())

    dataset_df.loc[:, columns.USER_ID_COLUMN] = dataset_df[columns.USER_ID_COLUMN].map(users_mapping)
    dataset_df.loc[:, columns.ARTIST_ID_COLUMN] = dataset_df[columns.ARTIST_ID_COLUMN].map(artists_mapping)

    return dataset_df, users_mapping, artists_mapping


def get_cv(dataset_df: pandas.DataFrame, test_size: int = 3) -> PredefinedSplit:
    items_ind_by_user = dataset_df.groupby(columns.USER_ID_COLUMN).cumcount(ascending=False)
    
    test_mask = items_ind_by_user < test_size

    test_fold = numpy.full(len(dataset_df), -1, dtype=numpy.int_)
    test_fold[test_mask] = 0

    return PredefinedSplit(test_fold)


def fit_and_eval(dataset_df: pandas.DataFrame,
                 pipeline_id: str) -> Pipeline:
    pipeline = pipelines[pipeline_id]
        
    X = dataset_df.loc[:, [columns.USER_ID_COLUMN, columns.ARTIST_ID_COLUMN]].to_numpy()

    logging.info('Got X, with shape %s', X.shape)

    cv = get_cv(dataset_df)

    logging.info('Got cv %s', cv)

    if pipeline_id in searchers:
        logging.info('Prepare searcher for "%s"', pipeline_id)

        searcher = searchers[pipeline_id][0](pipeline,
                                             cv=cv,
                                             **searchers[pipeline_id][1])
        search_result = searcher.fit(X)

        if search_result.best_params_:
            pipeline.set_params(**search_result.best_params_)

            logging.info('Search complete, find best params: %s', search_result.best_params_)
        else:
            raise ValueError('best_params_ not supported')
    else:
        logging.info('Got score: %s', cross_val_score(pipeline, X, scoring=..., cv=cv)[0])
    
    return pipeline.fit(X)


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

    dataset_df = download_dataframe(s3_client, constants.S3_BUCKET, constants.DATA_PATH / 'dataset.parquet')

    logging.info('Downloaded data, with shape %s', dataset_df.shape)

    dataset_df, users_mapping, artists_mapping = preprocess_dataset(dataset_df)

    logging.info('Dataset preprocessed, given %d users and %d artists',
                 len(users_mapping.id2code),
                 len(artists_mapping.id2code))
    
    logging.info('Fit pipeline "%s"', args.pipeline)

    pipeline = fit_and_eval(dataset_df, args.pipeline)

    logging.info('Given fitted pipeline:\n%s', pipeline)

    file_path = constants.SUBMISSION_PATH / args.pipeline / f'{run_dt.strftime("%Y%m%dT%H%M%S")}.parquet'

    # TODO Add code for submit (transform indexes from int to UUID)

    # upload_dataframe(submission_df, s3_client, constants.S3_BUCKET, file_path)

    logging.info('Recommendations saved to %s', file_path)
    logging.info('Done!')
