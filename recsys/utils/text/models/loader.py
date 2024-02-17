from .types import GensimPretraineWordModel, PretrainedWordModel

import gensim.downloader as gensim_api


class ModelLoader:
    def load(self, model_id: str, **kwargs) -> PretrainedWordModel:
        raise NotImplementedError


class GensimModelLoader(ModelLoader):
    def load(self, model_id: str) -> PretrainedWordModel:
        wv = gensim_api.load(model_id)
        return GensimPretraineWordModel(wv)
