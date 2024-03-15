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
                           PureSVDColabRecommender(random_state=constants.RANDOM_STATE))]),
    'funk_svd': Pipeline([('recommender',
                           FunkSVDColabRecommender(n_epochs=10,
                                                   random_state=constants.RANDOM_STATE,
                                                   verbose=True))]),
    'als_lightfm': Pipeline([('recommender',
                              LightFMColabRecommender(epochs=10,
                                                      num_threads=8,
                                                      verbose=True,
                                                      random_state=constants.RANDOM_STATE))]),
    'w2v': Pipeline([('recommender',
                      Word2VecColabRecommender(epochs=10,
                                               workers=5,
                                               seed=constants.RANDOM_STATE))]),
}


def score_wrapper(estimator: Pipeline, X: numpy.ndarray, y = None) -> float:
    if isinstance(estimator, Pipeline):
        logging.info('Evaluate')

        user_items = (
            pandas.DataFrame(X, columns=[columns.USER_ID_COLUMN, columns.ARTIST_ID_COLUMN])
            .groupby(columns.USER_ID_COLUMN)[columns.ARTIST_ID_COLUMN]
            .agg(lambda items: items.to_list())
        )

        y_true = user_items.to_numpy(dtype=numpy.object_)

        logging.info('Got y_true with shape %s', y_true.shape)

        y_pred = estimator.predict(user_items.index.to_numpy(), k=constants.AT_K)

        logging.info('Got y_pred with shape %s', y_pred.shape)

        return ndcg_score(y_true, y_pred, constants.AT_K)
    else:
        raise ValueError(estimator)


searchers = {
    'pure_svd': (
        OptunaSearchCV,
        {
            'param_distributions': {
                'recommender__n_components': CategoricalDistribution(choices=[16, 32, 64, 128]),
            },
            'n_jobs': 16,
            'n_trials': 50,
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
                'recommender__n_factors': CategoricalDistribution(choices=[16, 32, 64, 128]),
                'recommender__lr_all': FloatDistribution(low=0.001, high=0.01),
                'recommender__reg_all': FloatDistribution(low=0.001, high=10),
            },
            'n_jobs': 16,
            'n_trials': 50,
            'scoring': score_wrapper,
            'refit': False,
            'verbose': 4,
            'random_state': constants.RANDOM_STATE,
            'error_score': 'raise'
        }
    ),
    'als_lightfm': (
        OptunaSearchCV,
        {
            'param_distributions': {
                'recommender__no_components': CategoricalDistribution(choices=[16, 32, 64, 128]),
                'recommender__loss': CategoricalDistribution(choices=['logistic', 'bpr', 'warp']),
                'recommender__learning_rate': FloatDistribution(low=0.005, high=0.01),
                'recommender__item_alpha': FloatDistribution(low=0.05, high=0.1),
                'recommender__user_alpha': FloatDistribution(low=0.05, high=0.1),
            },
            'n_jobs': 16,
            'n_trials': 50,
            'scoring': score_wrapper,
            'refit': False,
            'verbose': 4,
            'random_state': constants.RANDOM_STATE,
            'error_score': 'raise'
        }
    ),
    'w2v': (
        OptunaSearchCV,
        {
            'param_distributions': {
                'recommender__sg': CategoricalDistribution(choices=[0, 1]),
                'recommender__window': IntDistribution(low=1, high=10),
                'recommender__ns_exponent': FloatDistribution(low=-3, high=3),
                'recommender__negative': IntDistribution(low=3, high=20),
                'recommender__min_count': IntDistribution(low=0, high=20),
                'recommender__vector_size': CategoricalDistribution(choices=[16, 32, 64, 128]),
            },
            'n_jobs': 16,
            'n_trials': 50,
            'scoring': score_wrapper,
            'refit': False,
            'verbose': 4,
            'random_state': constants.RANDOM_STATE,
            'error_score': 'raise'
        }
    ),
}
