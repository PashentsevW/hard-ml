import argparse
import logging
from datetime import datetime
from typing import Optional

import boto3
import numpy
import pandas
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

import columns
import constants
from config import pipelines, score_wrapper, searchers
from utils.io.s3 import download_dataframe, upload_dataframe


# def prepare_data_for_submit(pipeline: Pipeline,
#                             dataset_df: pandas.DataFrame) -> pandas.DataFrame:
    # users = dataset_df[columns.UID_COLUMN].unique()
    # items = pipeline.predict(users, k=constants.AT_K)

    # submission_df = pandas.DataFrame({
    #     columns.USER_ID_COLUMN: users
    # })

    # if items.ndim == 1:
    #     submission_df.loc[:, columns.Y_RECS_COLUMN] = (
    #         pandas.Series(items, index=submission_df.index)
    #         .apply(lambda row: numpy.array(row).flatten().tolist())
    #     )
    # else:
    #     submission_df.loc[:, columns.Y_RECS_COLUMN] = (
    #         pandas.DataFrame(items, index=submission_df.index)
    #         .apply(lambda row: numpy.array(row).flatten().tolist(), axis=1)
    #     )

    # return submission_df


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

    dataset_df = ...

    logging.info('Downloaded data, with shape %s', dataset_df.shape)

    logging.info('Fit pipeline "%s"', args.pipeline)

    pipeline = ...

    logging.info('Got fitted pipeline:\n%s', pipeline)
    
    file_path = constants.SUBMISSION_PATH / args.pipeline / f'{run_dt.strftime("%Y%m%dT%H%M%S")}.parquet'

    submission_df = ...

    logging.info('Got submission with shape %s and columns %s', 
                submission_df.shape,
                submission_df.columns)

    logging.info('Recommendations saved to %s', file_path)

    logging.info('Done!')
