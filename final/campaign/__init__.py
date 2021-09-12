import os
import argparse
from typing import Dict, List

from dask import dataframe as dd

from .source import Engine
from .featurize import compute_features


_datapath = '../data/'

_engine = Engine()
_engine.registerTable('campaigns', dd.read_csv(os.path.join(_datapath, 'campaigns.csv')))
_engine.registerTable('customers', dd.read_csv(os.path.join(_datapath, 'customers.csv')))
_engine.registerTable('receipts', dd.read_parquet(os.path.join(_datapath, 'receipts.parquet')))


def featurize(name: str, config: List[Dict]) -> None:
    features_dd = compute_features(config, _engine)
    features_dd.to_parquet(os.path.join(_datapath, f'{name}_features.parquet'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument()
