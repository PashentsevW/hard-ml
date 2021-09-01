import os

from dask import dataframe as dd

from .source import Engine


_datapath = '/root/hard-ml/final/data/'

_engine = Engine()
_engine.registerTable('campaigns', dd.read_csv(os.path.join(_datapath, 'campaigns.csv')))
_engine.registerTable('customers', dd.read_csv(os.path.join(_datapath, 'customers.csv')))
_engine.registerTable('receipts', dd.read_parquet(os.path.join(_datapath, 'receipts.parquet')))
