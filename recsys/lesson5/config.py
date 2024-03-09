import logging

import numpy
import pandas
from optuna.distributions import CategoricalDistribution, IntDistribution, FloatDistribution
from optuna.integration import OptunaSearchCV
from sklearn.pipeline import Pipeline

import columns
import constants
from utils.recommenders.colab import (FunkSVDColabRecommender,
                                      LightFMColabRecommender,
                                      PopularItemsColabRecommender, 
                                      PureSVDColabRecommender,
                                      Word2VecColabRecommender,)
from utils.validation.metrics import ndcg_score

pipelines = {
    'baseline': Pipeline([('recommender', PopularItemsColabRecommender())]),
    'pure_svd': Pipeline([('recommender',
                           PureSVDColabRecommender(n_components=5,
                                                   random_state=constants.RANDOM_STATE))]),
    'funk_svd': Pipeline([('recommender',
                           FunkSVDColabRecommender(n_factors=5,
                                                   n_epochs=1,
                                                   random_state=constants.RANDOM_STATE,
                                                   verbose=False))]),
    'als_lightfm': Pipeline([('recommender',
                              LightFMColabRecommender(no_components=128,
                                                      learning_rate=0.05,
                                                      user_alpha=5,
                                                      item_alpha=5,
                                                      epochs=40,
                                                      num_threads=10,
                                                      verbose=True,
                                                      random_state=constants.RANDOM_STATE))]),
    'w2v': Pipeline([('recommender',
                      Word2VecColabRecommender(sg=0,
                                               vector_size=32,
                                               min_count=18,
                                               negative=17,
                                               window=10,
                                               ns_exponent=0.05,
                                               epochs=30,
                                               workers=5,
                                               seed=42))]),
}


def score_wrapper(estimator: Pipeline, X: numpy.ndarray, y = None) -> float:
    if isinstance(estimator, Pipeline):
        logging.info('Evaluate')

        y_true = (
            pandas.DataFrame(X, columns=[columns.USER_ID_COLUMN, columns.ARTIST_ID_COLUMN])
            .groupby(columns.USER_ID_COLUMN)[columns.ARTIST_ID_COLUMN]
            .agg(lambda items: items.to_list())
            .to_numpy(dtype=numpy.object_)
        )

        logging.info('Got y_true with shape %s', y_true.shape)

        y_pred = estimator.predict(X, k=constants.AT_K)

        logging.info('Got y_pred with shape %s', y_pred.shape)

        return ndcg_score(y_true, y_pred, constants.AT_K)
    else:
        raise ValueError(estimator)


searchers = {
    'pure_svd': (
        OptunaSearchCV,
        {
            'param_distributions': {
                'recommender__n_components': IntDistribution(low=2, high=50),
            },
            'n_trials': 5,
            'scoring': score_wrapper,
            'refit': False,
            'verbose': 4,
            'random_state': constants.RANDOM_STATE,
            'error_score': 'raise'
        }
    ),
    'funk_svd': (
        OptunaSearchCV,
        {
            'param_distributions': {
                'recommender__n_factors': IntDistribution(low=2, high=200),
                'recommender__n_epochs': IntDistribution(low=1, high=50),
                'recommender__lr_all': FloatDistribution(low=0.001, high=0.01),
                'recommender__reg_all': FloatDistribution(low=0.001, high=10.),
            },
            'n_trials': 50,
            'scoring': score_wrapper,
            'refit': False,
            'verbose': 4,
            'random_state': constants.RANDOM_STATE,
            'error_score': 'raise'
        }
    ),
    # 'als_lightfm': (
    #     OptunaSearchCV,
    #     {
    #         'param_distributions': {
    #             'recommender__no_components': IntDistribution(low=10, high=200),
    #             'recommender__loss': CategoricalDistribution(choices=['logistic', 'bpr', 'warp']),
    #             'recommender__learning_rate': FloatDistribution(low=0.001, high=0.01),
    #             'recommender__item_alpha': FloatDistribution(low=0.05, high=5),
    #             'recommender__user_alpha': FloatDistribution(low=0.05, high=5),
    #         },
    #         'n_jobs': 16,
    #         'n_trials': 50,
    #         'scoring': score_wrapper,
    #         'refit': False,
    #         'verbose': 4,
    #         'random_state': constants.RANDOM_STATE,
    #         'error_score': 'raise'
    #     }
    # ),
    # 'w2v': (
    #     OptunaSearchCV,
    #     {
    #         'param_distributions': {
    #             'recommender__sg': CategoricalDistribution(choices=[0, 1]),
    #             'recommender__window': IntDistribution(low=1, high=10),
    #             'recommender__ns_exponent': FloatDistribution(low=-3, high=3),
    #             'recommender__negative': IntDistribution(low=3, high=20),
    #             'recommender__min_count': IntDistribution(low=0, high=20),
    #             'recommender__vector_size': CategoricalDistribution(choices=[16, 32, 64, 128]),
    #         },
    #         'n_jobs': 16,
    #         'n_trials': 50,
    #         'scoring': score_wrapper,
    #         'refit': False,
    #         'verbose': 4,
    #         'random_state': constants.RANDOM_STATE,
    #         'error_score': 'raise'
    #     }
    # ),
}
