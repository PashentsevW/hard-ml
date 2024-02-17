from typing import List, Any

import numpy as np


def user_hitrate(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: 1 if top-k recommendations contains at lease one relevant item
    """
    return int(len(set(y_rec[:k]).intersection(set(y_rel))) > 0)


def user_precision(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: percentage of relevant items through recommendations
    """
    return len(set(y_rec[:k]).intersection(set(y_rel))) / k


def user_recall(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: percentage of found relevant items through recommendations
    """
    return len(set(y_rec[:k]).intersection(set(y_rel))) / len(y_rel)


def user_ap(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: average precision metric for user recommendations
    """
    return ...


def user_ndcg(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: ndcg metric for user recommendations
    """
    dcg = np.where(np.isin(y_rec[:k], y_rel), 1 / np.log2(1 + np.arange(1, len(y_rec[:k]) + 1)), 0).sum()
    idcg = (1 / np.log2(1 + np.arange(1, min(len(y_rec[:k]), len(y_rel)) + 1))).sum()

    return dcg / idcg if idcg else 0


def user_rr(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: reciprocal rank for user recommendations
    """
    first_relevant_rank = np.where(np.isin(y_rec[:k], y_rel))[0]
    if len(first_relevant_rank) != 0:
        return 1 / (first_relevant_rank[0] + 1)
    else:
        return 0
