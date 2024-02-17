from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from utils.recommenders.content import (PopularItemsContentRecommender,
                                        RandomItemsContentRecommender,
                                        SimilarItemsContentRecommender)
from utils.validation.metrics import hitrate_score

import constants
import columns
from estimators import embedding_vectorizer, simple_vectorizer

pipelines = {
    'random': Pipeline([('recommender', RandomItemsContentRecommender(constants.RANDOM_STATE))]),
    'popular': Pipeline(
        [('popularity_transformer',
          make_column_transformer((SimpleImputer(strategy='constant', fill_value=-1),
                                   ['rating_value']),
                                  remainder='drop')),
         ('recommender', PopularItemsContentRecommender())]
    ),
    'keywords': Pipeline(
        [('keywords_vectorize', make_column_transformer((simple_vectorizer(), 'keywords'),
                                                        remainder='drop')),
         ('normalizer', Normalizer()),
         ('recommender', SimilarItemsContentRecommender())]
    ),
    'similarity': Pipeline(
        [('vectorizer',
          ColumnTransformer(
            transformers=[
                ('keywords', simple_vectorizer(), 'keywords'),
                ('stars', simple_vectorizer(), 'stars'),
                ('directors', simple_vectorizer(), 'directors'),
                ('creators', simple_vectorizer(), 'creators'),
                ('genre', simple_vectorizer(), 'genre'),
                ('description', embedding_vectorizer(), 'description'),
            ],
            remainder='drop')),
         ('recommender', SimilarItemsContentRecommender())]
    )
}


def score_wrapper(estimator, X, y) -> float:
    if isinstance(estimator, Pipeline):
        y_pred = (estimator[-1]
                  .predict(X[columns.TITLE_ID_COLUMN].to_numpy().reshape(-1, 1),
                           k=constants.AT_K))
        return hitrate_score(y, y_pred, constants.AT_K)
    else:
        raise ValueError(estimator)


searchers = {
    'similarity': (
        GridSearchCV,
        {'param_grid': {
            'vectorizer__description__embedding__word_model_id': ['glove-wiki-gigaword-50',],
            'recommender__distance_metric': ['cosine', 'l1', 'l2', 'euclidean'],
         },
         'scoring': score_wrapper,
         'refit': False,
         'verbose': 4,
         'error_score': 'raise'})
}
