from dask import dataframe as dd
from .base import FeatureCalcer, DateFeatureCalcer
from ..source import Engine


class PurchasesAggregateCalcer(DateFeatureCalcer):
    name = 'purchases_aggregate'
    keys = ['customer_id']

    def compute(self) -> dd.DataFrame:
        source_dd = self.engine.getTable('receipts')
        source_flt_dd = self.flt(source_dd)

        features_dd = (source_flt_dd
                       .groupby(self.keys)
                       .agg({'date': 'count',
                             'purchase_amt': 'sum',
                             'purchase_sum': 'sum',})
                       .reset_index())
        features_dd.columns = self.keys + [f'purchases_{self.delta}d__count',
                                           f'purchases_amt_{self.delta}d__sum',
                                           f'purchases_sum_{self.delta}d__sum',]
        return features_dd


class AgeLocationCalcer(FeatureCalcer):
    name = 'age_location'
    keys = ['customer_id']

    def compute(self) -> dd.DataFrame:
        source_dd = self.engine.getTable('customers')
        return source_dd[self.keys + ['age', 'location']]

