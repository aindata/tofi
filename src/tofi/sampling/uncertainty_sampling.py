import numpy as np


def least_confidence(proba: np.ndarray, issorted: bool = False) -> np.ndarray:
    """

    :param proba: Prediction probabilities of an instance has a dimension of 1 by number of classes
    :param issorted: checks whether proba is sorted
    :return: sampling score of prediction

    TODO: write latex formula for the implementation
    """

    if issorted:
        confidence = proba[0]
    else:
        confidence = np.max(proba)
    num_labels = proba.size

    normalized_confidence = (1 - confidence) * (num_labels / (num_labels - 1))
    return normalized_confidence


def margin_confidence(proba: np.ndarray, issorted: bool = False) -> np.ndarray:
    """

    :param proba: :param proba: Prediction probabilities of an instance has a dimension of 1 by number of classes
    :param issorted:
    :return:
    """
    sorted_proba = proba
    if not issorted:
        sorted_proba = np.sort(proba)[::-1]  # sort probs s.t. the largest is the first

    diff = (sorted_proba[0] - sorted_proba[1])  # difference between top two probabilities
    # TODO: why subtracting the top 2 probabilities?

    margin_conf = 1 - diff
    return margin_conf


def ratio_confidence(proba: np.ndarray, issorted: bool = False) -> np.ndarray:
    """

    :param proba: :param proba: Prediction probabilities of an instance has a dimension of 1 by number of classes
    :param issorted:
    :return:
    """
    sorted_proba = proba
    if not issorted:
        sorted_proba = np.sort(proba)[::-1]  # sort probs s.t. the largest is the first
    ratio = sorted_proba[1] / sorted_proba[0]  # ratio between top two pros
    # TODO: why dividing the top two probabilities? what if there are more than 2 classes?
    return ratio


def entropy_confidence(proba: np.ndarray) -> np.ndarray:
    """

    :param proba: :param proba: Prediction probabilities of an instance has a dimension of 1 by number of classes
    :return:
    """
    log_probs = proba * np.log2(proba)
    raw_entropy = 0 - np.sum(log_probs)
    normalized_entropy = raw_entropy / np.log2(proba.size)
    return normalized_entropy
