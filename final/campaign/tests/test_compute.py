import sys
import pytest
from dask import dataframe as dd
sys.path.append('/workspaces/Uplift/final')
from campaign.source import Engine
from campaign.featurize import compute_features


_engine = Engine()
_engine.registerTable('campaigns', dd.read_csv('final/data/campaigns.csv'))
_engine.registerTable('customers', dd.read_csv('final/data/customers.csv'))
_engine.registerTable('receipts', dd.read_parquet('final/data/receipts.parquet'))


def test_features_calcers():
    calcers_config = [{'name': 'purchases_aggregate',
                       'args': {'col_date': 'date',
                                'date_to': 102,
                                'delta': 20,}},
                      {'name': 'age_location',
                       'args': {}},]
    features_dd = compute_features(calcers_config, _engine)

    assert len(features_dd.columns) == 6

    