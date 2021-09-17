import os
import argparse
import pathlib
from typing import Dict, List

import yaml
from dask import dataframe as dd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import *

from .source import Engine
from .featurise import compute_features
from .estimators import build_pipeline


_engine = Engine()
_datapath = None


def _init(workpath: str) -> None:
    global _datapath
    _datapath = os.path.join(workpath, 'data')

    _engine.registerTable('campaigns', dd.read_csv(os.path.join(_datapath, 'campaigns.csv')))
    _engine.registerTable('customers', dd.read_csv(os.path.join(_datapath, 'customers.csv')))
    _engine.registerTable('receipts', dd.read_parquet(os.path.join(_datapath, 'receipts.parquet')))


def featurize(name: str, config: List[Dict]) -> None:
    features_dd = compute_features(config['calcers'], _engine)
    features_dd.to_parquet(os.path.join(_datapath, f'{name}_features.parquet'))


def train(name: str, config: List[Dict]) -> None:
    config = {'sample_frac': 0.1,
              'random_state': 110894,
              'transformers': [{'name': 'label_encoder',
                                'args': {'columns': ['location'], }}], 
              'selectors': [{'name': 'dummy_selector',
                             'args': dict()}],
              'model': {'name': 'uplift_random_forest',
                        'args': {'evaluationFunction': 'ED',
                                 'random_state': 110894,}}, }

    features_dd = dd.read_parquet(os.path.join(_datapath, f'{name}_features.parquet'))
    features_df = (features_dd
                   .sample(frac=config['sample_frac'],
                           random_state=config['random_state'])
                   .compute())

    X = features_df.loc[:, features_df.columns[3:]]
    w = features_df.loc[:, 'target_group_flag'].fillna(0)
    y = (28 * features_df.loc[:, 'target_purchase_amt'].fillna(0)
         - features_df.loc[:, 'target_discount_sum'].fillna(0)
         - 1 * features_df.loc[:, 'target_group_flag'].fillna(0))

    pipeline = Pipeline([('transform', build_pipeline(config['transformers'])),
                         ('select', build_pipeline(config['selectors'])),
                         ('model', build_pipeline([config['model']]))])
    print(pipeline)

    (X_train, X_test,
     y_train, y_test,
     w_train, w_test, ) = train_test_split(X, y, w,
                                           test_size=0.3,
                                           random_state=110894,
                                           stratify=w)
    print(X_train.shape, X_test.shape)

    pipeline.fit(X_train, y_train, model__w=w_train)
    uplift = pipeline.predict(X_test)
    print(uplift)


_tasks = {'featurize': featurize, 
          'train': train, }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--name', type=str, required=True)
    ap.add_argument('-t', '--task', type=str, required=True)
    ap.add_argument('-c', '--config', type=pathlib.Path, required=True)
    ap.add_argument('-d', '--work-dir', type=pathlib.Path, required=False, default='.')

    args = vars(ap.parse_args())

    _init(args['work_dir'])
    
    with open(args['config'], 'r') as f:
        config = yaml.load(f, yaml.Loader)
    
    _tasks[args['task']](args['name'], config)
