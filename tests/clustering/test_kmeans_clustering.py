import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import logging

import sys
sys.path.append('../..')  # TODO: find better way to import files from src
from src.tofi.clustering.kmeans_clustering import KMeansClusterSampler


# TODO: I have no idea on how to write some tests for some of the functions
# TODO: Questions on how to create test environment:
# How do we test the methods of an object, e.g. who initiates the object? me or automated tester?


def setup_data_environment():
    """
    Create a dataset and trained KMeans cluster with predefined seed values,
    all the asserts here indicate the fitted values here

    """
    X, y = make_blobs(n_samples=30, n_features=2, random_state=0)
    assert X.sum() == 90.7697879524822
    scikit_model = KMeans(n_clusters=3, max_iter=1000, random_state=0)
    scikit_model.fit(X)
    assert (np.array([0, 0, 0, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0, 1, 2, 2, 1, 2, 0, 2, 1, 1, 2, 2, 2, 1, 2, 1, 0, 2]) ==
            scikit_model.labels_).all()

    cl_centers = np.array([[-2.09299022, 2.88979044],
                           [2.03110596, 0.77298308],
                           [1.62725797, 4.04956044]])
    assert np.isclose(scikit_model.cluster_centers_, cl_centers).all()
    # Scikit-learn KMeans creates the following clusters with sorted order from their centroids
    # cluster0 sorted: [7, 28, 9, 2, 11, 1, 12, 10, 3, 0, 18]
    # cluster1 sorted: [16, 21, 8, 4, 25, 13, 20, 27, 6]
    # cluster2 sorted: [29, 14, 15, 22, 19, 5, 17, 23, 26, 24]
    logging.info('Data and scikit-learn model created')
    return X, scikit_model


def test_cluster_sorting_distance(sampler_model):
    """

    :param sampler_model: pre-trained clustering KMeans model
    """
    assert (sampler_model.cluster_to_instances_sorted[0] == [7, 28, 9, 2, 11, 1, 12, 10, 3, 0, 18]).all()
    assert (sampler_model.cluster_to_instances_sorted[1] == [16, 21, 8, 4, 25, 13, 20, 27, 6]).all()
    assert (sampler_model.cluster_to_instances_sorted[2] == [29, 14, 15, 22, 19, 5, 17, 23, 26, 24]).all()
    logging.info('Instance sorting test passed')

def test_get_centroid_samples(sampler_model):
    """
    Tests the closest sample method
    :param sampler_model: pre-trained clustering KMeans model
    """
    assert sampler_model.get_centroid_samples() == [7, 16, 29]
    logging.info('Centroid samples test passed')


def test_get_outlier_samples(sampler_model):
    """

    :param sampler_model:
    :return:
    """
    assert sampler_model.get_outlier_samples()
    logging.info('Outlier samples test passed')


def test_main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    X, scikit_model = setup_data_environment()
    sampler_model = KMeansClusterSampler(n_clusters=3,
                                         max_iter=100,
                                         random_state=0
                                         )
    sampler_model.fit(X)
    test_get_centroid_samples(sampler_model)
    test_get_outlier_samples(sampler_model)
    test_cluster_sorting_distance(sampler_model)


test_main()



