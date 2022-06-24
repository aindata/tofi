"""
Implementation of Diversity based sampling.
Resources:
1. Human-in-the-loop Machine Learning Book by Robert (Munro) Monarch
2. http://www.robertmunro.com/Diversity_Sampling_Cheatsheet.pdf
3. https://github.com/rmunro/pytorch_active_learning
"""

import numpy as np
from random import shuffle
from typing import List, Any


def _get_rank(unlabeled_logit: np.float, validation_logits: List[float]) -> float:
    """
    TODO: add docstring
    :param unlabeled_logit:
    :param validation_logits:
    :return:
    """

    idx = 0
    for ranked_number in validation_logits:
        if unlabeled_logit < ranked_number:
            break  # TODO: validation_logits sorted, hence implement binary search
        idx += 1

    if idx >= len(validation_logits):
        idx = len(validation_logits)
    elif idx > 0:
        diff = validation_logits[idx] - validation_logits[idx - 1]
        perc = unlabeled_logit - validation_logits[idx - 1]
        linear = perc / diff
        idx = float(idx - 1) + linear

    absolute_ranking = idx / len(validation_logits)
    return absolute_ranking


# TODO: model outliers currently only support neural models by employing the raw prediction outputs
# TODO: add different model support
def model_outliers(scores: List[Any],
                   validation_rankings: List[Any],
                   number: int,
                   ) -> List[Any]:
    """
    Get outliers from unlabeled data based on validation rankings generated from model output.

    An outlier is defined by Monarch as the lowest average from rank order of logits
    where rank order is defined by validation data inference

    Validation ranking can be generated from raw scores
    :param validation_rankings: creates neuron activation(s) for
    :param scores: model output scores of unlabeled data
    :param number: number of outlier instances to return
    :return: number of outlier instances
    """
    # algorithm
    # len(scores) denotes the number of unlabeled data
    # len(score) denotes the number of neurons in model
    # 1. generate rank for each score neuron in scores by _get_rank e.g. 2 ranks for binary
    # 2. generate average rank by 1 - (sum(ranks) / len(score)) for each score
    # 3. sort rankings of scores and return the least ranked (biggest) instances
    pass


def cluster_based():
    pass


def representative(adaptive: bool = False):
    pass

