import os
import argparse
import pathlib
from typing import Dict, List

import yaml
from dask import dataframe as dd
from sklearn.pipeline import Pipeline

from .source import Engine
from .featurize import compute_features
from .estimators import build_pipeline


_datapath = 'data/'

_engine = Engine()
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
                             'args': dict()}], }

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
                         ('select', build_pipeline(config['selectors']))])
    



_tasks = {'featurize': featurize, 
          'train': train, }

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--name', type=str, required=True)
    ap.add_argument('-t', '--task', type=str, required=True)
    ap.add_argument('-c', '--config', type=pathlib.Path, required=True)

    args = vars(ap.parse_args())
    
    with open(args['config'], 'r') as f:
        config = yaml.load(f, yaml.Loader)
    
    _tasks[args['task']](args['name'], config)
