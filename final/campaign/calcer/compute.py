from typing import List, Dict
from dask import dataframe as dd
from .base import FeatureCalcer
from .calcers import *
from ..source import Engine


_calcers = dict()


def registerCalcer(class_calcer) -> None:
    _calcers[class_calcer.name] = class_calcer


def _createCalcer(name: str, **kwargs) -> FeatureCalcer:
    return _calcers[name](**kwargs)


def _join_tables(tables: List[dd.DataFrame], on: List[str], how: str) -> dd.DataFrame:
    result = tables[0]
    for table in tables[1: ]:
        result = result.merge(table, on=on, how=how)
    return result


def compute_features(config: List[Dict], engine: Engine) -> dd.DataFrame:
    calcers = list()
    keys = None

    for calcer_config in config:
        name, args = tuple(calcer_config.values())
        args['engine'] = engine

        calcer = _createCalcer(name, **args)

        if keys is None:
            keys = calcer.keys
        
        if (keys is not None 
            and keys != calcer.keys):
            raise ValueError(calcer.keys)

        calcers.append(calcer)

    compute_results = list()
    for calcer in calcers:
        compute_results.append(calcer.compute())

    if not isinstance(keys, list):
        keys = [keys]

    return _join_tables(compute_results, on=keys, how='outer')

