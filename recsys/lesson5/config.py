from sklearn.pipeline import Pipeline

from utils.recommenders.colab import PopularItemsColabRecommender

pipelines = {
    'baseline': Pipeline([('recommender', PopularItemsColabRecommender())]),
}

searchers = {}
