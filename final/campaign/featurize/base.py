from abc import ABC, abstractmethod
from dask import dataframe as dd
from ..source import Engine


_seed = 110894


class FeatureCalcer(ABC):
    name = None
    keys = None

    def __init__(self, engine: Engine):
        self.engine = engine

    @abstractmethod
    def compute(self) -> dd.DataFrame:
        pass


class DateFeatureCalcer(FeatureCalcer):
    def __init__(self,
                 engine: Engine,
                 col_date: str,
                 date_to: int,
                 delta: int):
        super().__init__(engine)
        self.col_date = col_date
        self.date_to = date_to
        self.delta = delta

    def flt(self, data: dd.DataFrame) -> dd.DataFrame:
        date_to = self.date_to
        date_from = self.date_to - self.delta
        return data[((data[self.col_date] >= date_from)
                     & (data[self.col_date] < date_to))]


class TargetCalcer(FeatureCalcer):
    def __init__(self,
                 engine: Engine, 
                 sample_frac: float = None):
        super().__init__(engine)
        self.sample_frac = sample_frac

    def collect(self) -> dd.DataFrame:
        raise NotImplementedError
    
    def compute(self) -> dd.DataFrame:
        if self.sample_frac is None:
            return self.collect()
        return (self.collect()
                .sample(frac=self.sample_frac,
                        random_state=_seed))

