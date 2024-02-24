import argparse
import logging
from datetime import datetime

import boto3

import constants
from config import pipelines, searchers
from utils.io.s3 import download_dataframe, upload_dataframe


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

    # TODO Add preparing data: transform UUID to int
    
    logging.info('Fit pipeline "%s"', args.pipeline)

    # TODO Add code for fit
    # TODO Add code for eval

    file_path = constants.SUBMISSION_PATH / args.pipeline / f'{run_dt.strftime("%Y%m%dT%H%M%S")}.parquet'

    # TODO Add code for submit (transform indexes from int to UUID)

    # upload_dataframe(submission_df, s3_client, constants.S3_BUCKET, file_path)

    logging.info('Recommendations saved to %s', file_path)
    logging.info('Done!')
