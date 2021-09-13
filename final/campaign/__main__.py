import os
import argparse
import pathlib
from typing import Dict, List

import yaml
from dask import dataframe as dd

from .source import Engine
from .featurize import compute_features


_datapath = 'data/'

_engine = Engine()
_engine.registerTable('campaigns', dd.read_csv(os.path.join(_datapath, 'campaigns.csv')))
_engine.registerTable('customers', dd.read_csv(os.path.join(_datapath, 'customers.csv')))
_engine.registerTable('receipts', dd.read_parquet(os.path.join(_datapath, 'receipts.parquet')))


def featurize(name: str, config: List[Dict]) -> None:
    features_dd = compute_features(config['calcers'], _engine)
    features_dd.to_parquet(os.path.join(_datapath, f'{name}_features.parquet'))


_tasks = {'featurize': featurize, }

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--name', type=str, required=True)
    ap.add_argument('-t', '--task', type=str, required=True)
    ap.add_argument('-c', '--config', type=pathlib.Path, required=True)

    args = vars(ap.parse_args())
    
    with open(args['config'], 'r') as f:
        config = yaml.load(f, yaml.Loader)
    
    _tasks[args['task']](args['name'], config)
