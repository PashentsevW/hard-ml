from inspect import ArgSpec
from typing import Dict, List
from sklearn.pipeline import Pipeline

from .transformers import *
from .selectors import *
from .models import *


_estimators = {}
for estimator in [LabelEncoder,
                  Imputer,
                  DummySelector, ]:
    _estimators[estimator.name] = estimator

_estimators['uplift_random_forest'] = UpliftRandomForest


def build_pipeline(config: List[Dict]) -> Pipeline:
    steps = list()
    for item in config:
        name, args = tuple(item.values())
        steps.append((name, _estimators[name](**args)))
    if len(steps) == 1:
        return steps[0][1]
    return Pipeline(steps)
