from inspect import ArgSpec
from typing import Dict, List
from sklearn.pipeline import Pipeline

from .transformers import *
from .selectors import *


_estimators = {}
for estimator in [LabelEncoder,
                  DummySelector, ]:
    _estimators[estimator.name] = estimator


def build_pipeline(config: List[Dict]) -> Pipeline:
    steps = list()
    for item in config:
        name, args = tuple(item.values())
        steps.append((name, _estimators[name](**args)))
    return Pipeline(steps)
