import datetime
import dask.dataframe as dd
import pandas as pd
import numpy as np
try:
    import data_config as cfg
except ImportError:
    pass
from featurelib import DateFeatureCalcer, Engine


class DayOfWeekReceiptsCalcer(DateFeatureCalcer):
    name = 'day_of_week_receipts'
    keys = ['client_id']

    def __init__(self,
                 engine: Engine, 
                 date_to: datetime.date,
                 delta: int,
                 **kwargs):
        super().__init__(engine=engine,
                         date_to=date_to,
                         **kwargs)

        self.engine = engine
        self.delta = delta

    def compute(self) -> dd.DataFrame:
        source_dd = self.engine.get_table('receipts')
        date_to = pd.to_datetime(self.date_to)
        date_from = pd.to_datetime(self.date_to - datetime.timedelta(days=self.delta))

        mask = ((source_dd['transaction_datetime'] < date_to)
                & (source_dd['transaction_datetime'] >= date_from))
        source_flt_dd = source_dd[mask]

        table_tmp_dd = source_flt_dd.copy()
        table_tmp_dd['day_of_week'] = (source_flt_dd['transaction_datetime']
                                       .dt.dayofweek
                                       .astype('category')
                                       .cat
                                       .as_known())

        features_dd = table_tmp_dd.pivot_table(index='client_id',
                                               columns='day_of_week',
                                               values='transaction_id',
                                               aggfunc='count')
        features_dd.columns = [f'purchases_count_dw{f}__{self.delta}d' 
                               for f in features_dd.columns]
        features_dd = features_dd.reset_index()

        return features_dd


class FavouriteStoreCalcer(DateFeatureCalcer):
    name = 'favourite_store'
    keys = ['client_id']

    def __init__(self,
                 engine: Engine, 
                 date_to: datetime.date,
                 delta: int,
                 **kwargs):
        super().__init__(engine=engine,
                         date_to=date_to,
                         **kwargs)

        self.engine = engine
        self.delta = delta

    def compute(self) -> dd.DataFrame:
        source_dd = self.engine.get_table('receipts')
        date_to = pd.to_datetime(self.date_to)
        date_from = pd.to_datetime(self.date_to - datetime.timedelta(days=self.delta))

        mask = ((source_dd['transaction_datetime'] < date_to)
                & (source_dd['transaction_datetime'] >= date_from))
        source_flt_dd = source_dd[mask]

        def get_mode(x):
            values, counts = x.values[0]
            argmax = counts.argmax()
            if (counts == counts[argmax]).sum() > 1:
                return values[counts == counts[argmax]].max()
            else:
                return values[counts.argmax()]

        mode = dd.Aggregation('mode',
                              lambda x: x.apply(lambda y: np.unique(y, return_counts=True)),
                              lambda x: x.apply(lambda y: get_mode(y)),)
        features_dd = source_flt_dd.groupby('client_id').agg({'store_id': mode})
        features_dd.columns = [f'favourite_store_id__{self.delta}d']
        features_dd = features_dd.reset_index()

        return features_dd


if __name__ == '__main__':
    engine = Engine(tables={})
    engine.register_table(table=dd.read_parquet('task4/data/purchases.parquet'),
                          name='purchases')
    engine.register_table(table=dd.read_parquet('task4/data/receipts.parquet'),
                          name='receipts')
    engine.register_table(table=dd.read_csv('task4/data/products.csv'),
                          name='products')
    engine.register_table(table=dd.read_csv('task4/data/client_profile.csv'),
                          name='client_profile')
    engine.register_table(table=dd.read_csv('task4/data/campaigns.csv'),
                          name='campaigns')

    test_df = pd.read_csv('task4/dataset_mini.csv')
    
    # test calcers
    calcers = {item.name: item 
               for item in [DayOfWeekReceiptsCalcer,
                            FavouriteStoreCalcer,]}
    print('-' * 100)
    for calcer_cfg in cfg.data_config['calcers']:
        name, args = calcer_cfg.values()
        if calcers.get(name) is None:
            continue
        calcer = calcers[name](engine, **args)
        result_df = calcer.compute().compute()

        assert np.allclose(result_df.values,
                           test_df[result_df.columns].values,
                           atol=1e-5)
        
        print(f'Result of {name} calcer \nwith params {args}')
        print('-' * 100)
        print(result_df.head())
        print('-' * 100)

