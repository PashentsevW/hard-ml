import os
import argparse
import pathlib
from typing import Dict, List

import yaml
import pickle
import numpy as np
import pandas as pd
from dask import dataframe as dd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import *

from .source import Engine
from .featurise import compute_features
from .estimators import build_pipeline
from .metrics import uplift_at_k


_engine = Engine()
_datapath = None
_metricspath = None
_artifactspath = None


def _init(workpath: str) -> None:
    global _datapath
    _datapath = os.path.join(workpath, 'data')

    global _metricspath
    _metricspath = os.path.join(workpath, 'metrics')

    global _artifactspath
    _artifactspath = os.path.join(workpath, 'artifacts')

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

    pipeline.fit(X_train, y_train, model__w=w_train)

    with open(os.path.join(_artifactspath, f'{name}_pipeline.pkl'), 'wb') as f:
        pickle.dump(pipeline, f)

    uplift = pipeline.predict(X_test)

    hist, edges = np.histogram(uplift, bins=50)
    hist_ss = pd.Series(hist,
                        index=pd.IntervalIndex.from_arrays(left=edges[:-1],
                                                           right=edges[1:],
                                                           closed='left'))

    cutoff_step = 0.05 
    cutoffs = np.arange(cutoff_step, 1, cutoff_step, dtype=np.float16)
    metrics = np.array([uplift_at_k(uplift, w.values, y.values, k) 
                        for k in cutoffs])
    metrics_ss = pd.Series(metrics, index=cutoffs)

    example_df = X.copy()
    example_df['sample'] = np.nan
    example_df.loc[X_train.index, 'sample'] = 'train'
    example_df.loc[X_test.index, 'sample'] = 'test'
    example_df['w'] = w
    example_df['y'] = y
    example_df['uplift'] = pipeline.predict(X)

    hist_ss.to_csv(os.path.join(_metricspath, f'{name}_hist.csv'))
    metrics_ss.to_csv(os.path.join(_metricspath, f'{name}_metrics.csv'))
    example_df.to_csv(os.path.join(_metricspath, f'{name}_examples.csv'))


_tasks = {'featurise': featurize, 
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
