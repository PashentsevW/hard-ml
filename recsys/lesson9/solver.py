import argparse
import logging
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import boto3
import numpy
import pandas
from sklearn.pipeline import Pipeline

from utils.io.s3 import download_dataframe, download_object, upload_dataframe, upload_object

import columns
import constants
from config import (cand_data_pipelines,
                    cand_score_wrapper,
                    cand_pipelines,
                    rank_data_pipelines,
                    rank_score_wrapper,
                    rank_pipelines,
                    searchers,
                    splitter)
from estimators import SearcherBase, SplitterBase

ScoreFunctionType = Callable[[Pipeline, numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]], float]


def _prepare_data_for_cand_pipeline(sample_df: pandas.DataFrame, data_pipeline: str) -> numpy.ndarray:
    pass


def _prepare_data_for_rank_pipeline(
        train: bool,
        sample_df: pandas.DataFrame,
        data_pipelines: List[str]
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
    pass


def _prepare_pipeline(candidate: bool,
                      pipeline_id: str,
                      scorer: ScoreFunctionType,
                      cv: SplitterBase) -> Tuple[Pipeline, ScoreFunctionType, Optional[SearcherBase]]:
    pipeline = cand_pipelines[pipeline_id] if candidate else rank_pipelines[pipeline_id]
    searcher = None

    if pipeline_id in searchers:
        logging.info('Prepare searcher for "%s"', pipeline_id)

        searcher = searchers[pipeline_id][0](pipeline,
                                             cv=cv,
                                             **searchers[pipeline_id][1])

    return pipeline, scorer, searcher


def _train_and_eval_pipeline(pipeline: Pipeline,
                             score: ScoreFunctionType,
                             X_train: numpy.ndarray,
                             X_test: numpy.ndarray,
                             y_train: Optional[numpy.ndarray] = None,
                             y_test: Optional[numpy.ndarray] = None,
                             group_train: Optional[numpy.ndarray] = None,
                             group_test: Optional[numpy.ndarray] = None,
                             searcher: Optional[SearcherBase] = None) -> Pipeline:
    if searcher:
        search_result = searcher.fit(X_train, y_train, group=group_train)

        if hasattr(search_result, 'best_params_'):
            pipeline.set_params(**search_result.best_params_)

            logging.info('Search complete, find best params: %s', search_result.best_params_)
        else:
            raise ValueError('best_params_ not supported')

    pipeline.fit(X_train, y_train, group=group_train)

    logging.info('Got score on test sample', score(pipeline, X_test, y_test, group_test))

    X_full = numpy.vstack([X_train, X_test])
    y_full = numpy.hstack([y_train, y_test])
    group_full = numpy.hstack([group_train, group_test])

    pipeline.fit(X_full, y_full, group=group_full)

    return pipeline    


def train(sample_df: pandas.DataFrame,
          splitter: SplitterBase,
          cand_pipeline_id: str,
          cand_data_pipeline_id: str,
          cand_splitter: SplitterBase,
          rank_pipeline_id: Optional[str] = None,
          rank_data_pipeline_ids: Optional[List[str]] = None,
          rank_splitter: Optional[SplitterBase] = None,) -> Tuple[Pipeline, Pipeline]:
    train_idx, test_idx = (
        next(splitter
             .split(sample_df.loc[:, [columns.USER_ID_COLUMN,
                                      columns.ITEM_ID_COLUMN,
                                      columns.DT_COLUMN,
                                      columns.TARGET_COLUMN]]))
    )

    cand_X_train = _prepare_data_for_cand_pipeline(sample_df.loc[train_idx, :],
                                                   cand_data_pipeline_id)

    logging.info('Got train data for cand pipeline with shape: %s', cand_X_train.shape)

    cand_X_test = _prepare_data_for_cand_pipeline(sample_df.loc[test_idx, :],
                                                  cand_data_pipeline_id)

    logging.info('Got test data for cand pipeline with shape: %s', cand_X_test.shape)

    (cand_pipeline,
     cand_scorer,
     cand_searcher) = _prepare_pipeline(True, cand_pipeline_id, cand_score_wrapper, cand_splitter)

    cand_pipeline = _train_and_eval_pipeline(cand_pipeline,
                                             cand_scorer,
                                             cand_X_train, cand_X_test,
                                             None, None,
                                             None, None,
                                             cand_searcher)

    logging.info('Got fitted cand pipeline: %s', cand_pipeline)

    if not rank_pipeline_id:
        rank_pipeline = rank_pipelines['baseline']
    else:
        rank_X_train, rank_y_train, rank_group_train = (
            _prepare_data_for_rank_pipeline(True,
                                            sample_df.loc[train_idx, :],
                                            rank_data_pipeline_ids or ['baseline'])
        )

        logging.info('Got train data for rank pipeline: features %s; target %s; group %s',
                     rank_X_train.shape,
                     rank_y_train.shape,
                     rank_group_train.shape)

        rank_X_test, rank_y_test, rank_group_test = (
            _prepare_data_for_rank_pipeline(True,
                                            sample_df.loc[test_idx, :],
                                            rank_data_pipeline_ids or ['baseline'])
        )

        logging.info('Got test data for rank pipeline: features %s; target %s; group %s',
                    rank_X_test.shape,
                    rank_y_test.shape,
                    rank_group_test.shape)
        
        (rank_pipeline,
         rank_scorer,
         rank_searcher) = _prepare_pipeline(False, rank_pipeline_id, rank_score_wrapper, rank_splitter)

        rank_pipeline = _train_and_eval_pipeline(rank_pipeline,
                                                 rank_scorer,
                                                 rank_X_train, rank_X_test,
                                                 rank_y_train, rank_y_test,
                                                 rank_group_train, rank_group_test,
                                                 rank_searcher)

    logging.info('Got fitted rank pipeline: %s', rank_pipeline)

    recommendations_df = inference(sample_df.loc[test_idx, columns.USER_ID_COLUMN].drop_duplicates(),
                                   cand_pipeline,
                                   rank_pipeline,
                                   rank_data_pipeline_ids)

    logging.info('Got recommendations for test users with shape %s', recommendations_df.shape)

    # TODO add code for calculate ndcg

    logging.info('Got cand+rank score for test sample: %d', ...)

    return cand_pipeline, rank_pipeline


def inference(sample_df: pandas.DataFrame,
              cand_pipeline: Pipeline,
              rank_pipeline: Pipeline,
              rank_data_pipeline_ids: List[str]) -> pandas.DataFrame:
    users = sample_df[columns.USER_ID_COLUMN].to_numpy()

    logging.info('Got %d users', len(users))

    candidates = cand_pipeline.predict(users, k=...)

    logging.info('Got candidates with shape', len(candidates))

    rank_sample_df = (pandas.DataFrame({columns.USER_ID_COLUMN: users,
                                        columns.ITEM_ID_COLUMN: candidates})
                      .explode(columns.ITEM_ID_COLUMN))

    logging.info('Got sample for rank with shape', rank_sample_df.shape)
    
    rank_features, _, _ = _prepare_data_for_rank_pipeline(
        False,
        rank_sample_df,
        rank_data_pipeline_ids
    )
    rank_sample_df[columns.RELEVANCE_COLUMN] = rank_pipeline.predict(rank_features)

    logging.info('Got item`s relevances from ranker')

    recommedations_df = (rank_sample_df
                         .sort_values([columns.USER_ID_COLUMN, columns.RELEVANCE_COLUMN], ascending=[True, False])
                         .groupby(columns.USER_ID_COLUMN, as_index=False)
                         .agg({columns.ITEM_ID_COLUMN: lambda items: numpy.array(items).flatten().tolist()})
                         .rename(columns={columns.ITEM_ID_COLUMN: columns.Y_REC_COLUMN}))

    return recommedations_df


if __name__ == '__main__':
    run_dt = datetime.now()
    
    argparser = argparse.ArgumentParser('lesson9')
    argparser.add_argument('-m', '--mode',
                           type=str,
                           choices=['train', 'inference'],
                           required=True,
                           dest='mode')
    argparser.add_argument('-s', '--splitter',
                           type=str,
                           choices=splitter.keys(),
                           required=True,
                           dest='splitter')
    argparser.add_argument('--pipeline-cand',
                           type=str,
                           required=True,
                           dest='pipeline_cand')
    argparser.add_argument('--pipeline-cand-data',
                           type=str,
                           choices=cand_data_pipelines.keys(),
                           required=False,
                           dest='pipeline_cand_data')
    argparser.add_argument('--pipeline-cand-splitter',
                           type=str,
                           choices=splitter.keys(),
                           required=False,
                           dest='pipeline_cand_splitter')
    argparser.add_argument('--pipeline-rank',
                           type=str,
                           required=False,
                           dest='pipeline_rank')
    argparser.add_argument('--pipeline-rank-data',
                           type=str,
                           choices=rank_data_pipelines.keys(),
                           nargs='+',
                           required=False,
                           dest='pipelines_rank_data')
    argparser.add_argument('--pipeline-rank-splitter',
                           type=str,
                           choices=splitter.keys(),
                           default='dummy',
                           required=False,
                           dest='pipeline_rank_splitter')
    argparser.add_argument('-v', '--verbose',
                           default='INFO',
                           type=lambda arg: logging.getLevelName(arg),
                           required=False,
                           dest='loglevel')

    args = argparser.parse_args()

    logging.basicConfig(level=args.loglevel)
    logging.debug(args.__dict__)

    s3_session = boto3.session.Session()
    s3_client = s3_session.client(service_name='s3', endpoint_url='https://storage.yandexcloud.net')

    if args.mode == 'train':
        sample_df = download_dataframe(s3_client, constants.S3_BUCKET, constants.DATA_PATH / 'train.parquet')

        logging.info('Got train sample with shape: %s', sample_df.shape)

        cand_pipeline, rank_pipeline = train(sample_df,
                                             splitter[args.splitter],
                                             args.pipeline_cand,
                                             args.pipeline_cand_data,
                                             splitter[args.pipeline_cand_splitter],
                                             args.pipeline_rank,
                                             args.pipelines_rank_data,
                                             splitter.get(args.pipeline_rank_splitter))
        
        file_path = constants.ARTIFACTS_PATH / 'cand' / f'{run_dt.strftime("%Y%m%dT%H%M%S")}.bin'
        upload_object(cand_pipeline, s3_client, constants.S3_BUCKET, file_path)

        logging.info('Candidate pipeline saved to %s', file_path)

        file_path = constants.ARTIFACTS_PATH / 'rank' / f'{run_dt.strftime("%Y%m%dT%H%M%S")}.bin'
        upload_object(rank_pipeline, s3_client, constants.S3_BUCKET, file_path)

        logging.info('Ranking pipeline saved to %s', file_path)
    elif args.mode == 'inference':
        sample_df = download_dataframe(s3_client, constants.S3_BUCKET, constants.DATA_PATH / 'test.parquet')

        logging.info('Got sample with shape: %s', sample_df.shape)

        file_path = constants.ARTIFACTS_PATH / 'cand' / f'{args.pipeline_cand}.bin'
        cand_pipeline: Pipeline = download_object(s3_client, constants.S3_BUCKET, file_path)

        logging.info('Got candidate pipeline: %s', cand_pipeline)

        file_path = constants.ARTIFACTS_PATH / 'rank' / f'{args.pipeline_cand}.bin'
        rank_pipeline: Pipeline = download_object(s3_client, constants.S3_BUCKET, file_path)

        logging.info('Got ranking pipeline: %s', rank_pipeline)

        submission_df = inference(sample_df, cand_pipeline, rank_pipeline, args.pipelines_data_rank)

        logging.info('Got submission with shape %s and columns %s', 
                     submission_df.shape,
                     submission_df.columns)

        file_path = constants.SUBMISSION_PATH / f'{run_dt.strftime("%Y%m%dT%H%M%S")}.parquet'
        upload_dataframe(submission_df, s3_client, constants.S3_BUCKET, file_path)

        logging.info('Recommendations saved to %s', file_path)
    else:
        raise ValueError('Unsupported mode: %s', args.mode)

    logging.info('Done!')
