"""
Cluster based sampling algorithms, please note that this section is genuinely adapted from
Human-in-the-loop book by Robert (Munro) Monarch

3 ways to sample from clusters:
 - 1. Random: Randomly sampling items from within each cluster
 - 2. Centroids: Sampling the centroids of clusters to represent core significant trend
 - 3. Outliers: Sampling the farthest items from cluster centers to find potentially interesting data

Monarch suggests that it's likely that outliers are rarely seen examples, hence sample them by only a small
number
"""

from abc import ABC

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.datasets import make_blobs

from .clustering import AbstractCluster
from .base_errors import NotEnoughPointsError, ClusterDataAlreadySet


class KMeansClusterSampler(AbstractCluster, ABC):
    """
    Implements KMeans clustering algorithm to extract cluster based samples
    """
    def __init__(self,
                 n_clusters=8,
                 max_features=None,
                 distance_metric='cosine',
                 max_iter=1000,
                 **kwargs
                 ):
        """

        :param n_clusters: number of clusters to sample from
        :param max_features: TODO
        :param distance_metric: cosine similarity
        :param max_iter: TODO
        :param kwargs: TODO
        """
        self.n_far = 1
        self.n_closer = 1
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.model = KMeans(n_clusters=n_clusters,
                            max_iter=max_iter,
                            **kwargs)
        self._X = None
        self.X_pca = None
        self.cluster_to_instances_sorted = {}  # holds cluster_id: [instance ids] in sorted order

    def fit(self, data):
        """
        calls fit method of KMeans from sckit-learn
        :param data:
        :return:
        """
        if data.shape[0] < self.n_clusters:
            raise NotEnoughPointsError(data_size=data.shape[0], n_clusters=self.n_clusters)
        self.set_x(data)

        if self.max_features:
            self.X_pca = PCA(n_components=self.max_features).fit_transform(self.get_x())
        # TODO: do we need to do self.model = ... thing here?
        self.model = self.model.fit(self.X_pca if self.X_pca is not None else self.get_x())

    def set_x(self, data):
        if self._X:
            raise ClusterDataAlreadySet
        self._X = data

    def get_x(self):
        return self._X

    def set_n_closer(self, n_closer):
        self.n_closer = n_closer

    def set_n_far(self, n_far):
        self.n_far = n_far

    def _get_centroids(self):
        """
        :return: cluster center coordinates
        """
        return self.model.cluster_centers_

    def get_labels(self):
        return self.model.labels_

    def get_instances_from_cluster(self, cluster_id: int):
        """
        Extracts instances belonging to cluster_id from data
        """
        instance_ids = np.where(self.get_labels() == cluster_id)[0]
        return instance_ids, self._X[instance_ids]

    def _cache_sorted_distances(self,
                                argsorted_distances: np.array,
                                cluster_id: int
                                ):
        self.cluster_to_instances_sorted[cluster_id] = argsorted_distances

    def _calculate_cluster_distances(self):
        """
        Computes the distance between cluster-centroid and each instance from that cluster
        """
        centroids = self._get_centroids()
        for cluster_id, centroid in enumerate(centroids):
            c_instance_ids, c_instance_vecs = self.get_instances_from_cluster(cluster_id)
            c_i_centroid = np.vstack([c_instance_vecs, centroid])
            # distance_matrix[i,j] denotes the distance between row_i and row_j
            # the last row of distance matrix holds the distances from cluster centroid to other instances
            # TODO: there are too many unnecessary computations, i.e. we only use the last row of distance_matrix
            distance_matrix = pairwise_distances(c_i_centroid, metric=self.distance_metric)
            # self._cache_pairwise_distances(distance_matrix[:-1, -1], cluster_id)
            argsorted_distances = c_instance_ids[np.argsort(distance_matrix[:-1, -1])]
            self._cache_sorted_distances(argsorted_distances, cluster_id)

    def get_close_samples_to_cluster(self, n_samples=0, cluster_id=None):
        """
        Computes n_samples many close samples to the cluster_id
        :param n_samples: number of samples to generate
        :param cluster_id: denotes the cluster work from
        :return: indexes of items that are closer
        """
        if n_samples != 1:
            self.set_n_closer(n_samples)
        if not self.cluster_to_instances_sorted:
            self._calculate_cluster_distances()
        return self.cluster_to_instances_sorted[cluster_id][:n_samples]

    def get_far_samples_to_cluster(self, n_samples=0, cluster_id=None):
        """
        Computes n_samples many samples that are farest from the clustercentroid
        :param n_samples: number of samples to generate
        :param cluster_id: denotes the cluster work from
        :return: indexes of items that are far
        """
        if n_samples != 1:
            self.set_n_far(n_samples)
        if not self.cluster_to_instances_sorted:
            self._calculate_cluster_distances()
        return self.cluster_to_instances_sorted[cluster_id][-n_samples:]

    def get_centroid_samples(self):
        """
        Computes the closest instance for each cluster
        :return: indexes of items that are closer
        """
        ret = []
        for cluster_id in range(self.n_clusters):
            ret += list(self.get_close_samples_to_cluster(1, cluster_id=cluster_id))
        return ret

    def get_outlier_samples(self):
        """
        Computes the farest instance for each cluster
        :return: indexes of items that are farest
        """
        ret = []
        for cluster_id in range(self.n_clusters):
            ret += list(self.get_far_samples_to_cluster(1, cluster_id=cluster_id))
        return ret

    def get_random_samples(self, n_samples: int):
        """
        Randomly sample from each cluster by equally,
        if there is a remainder randomly choose a cluster and sample from this cluster for remaining many items
        :param n_samples: number of samples to sample from
        :return: index of items that are randomly selected
        """
        n_samples_cluster = n_samples // self.n_clusters
        n_close, n_far = self.n_closer, self.n_far
        ret = []
        for cluster_id in range(self.n_clusters):
            temp = list(self.cluster_to_instances_sorted[cluster_id][n_close:-n_far])
            ret += list(np.random.choice(temp, n_samples_cluster, replace=False))
        if n_samples % self.n_clusters != 0:
            cluster_id = np.random.choice(self.n_clusters)
            temp = list(self.cluster_to_instances_sorted[cluster_id][n_close:-n_far])
            # this is a tricky set solution to assure uniqueness of remaining items
            to_sample = list(set(temp) - set(ret))
            ret += list(np.random.choice(to_sample, n_samples % self.n_clusters))
        return ret

    def get_samples(self, n_samples_random=None):
        """
        TODO: decide number of sampling strategy
        :param n_samples_random: number of random samples to be used
        :return:
        """
        if not n_samples_random:
            raise ValueError("You must specify number of samples")
        centroid_samples = self.get_centroid_samples()
        outlier_samples = self.get_outlier_samples()
        random_samples = self.get_random_samples(n_samples=n_samples_random)
        return centroid_samples + outlier_samples + random_samples


# EXAMPLE OF USAGE
# X, y = make_blobs(n_samples=30, n_features=2, random_state=0)
# assert X.sum() == 90.7697879524822
#
#
# sampler = KMeansClusterSampler(n_clusters=3,
#                                max_iter=1000,
#                                random_state=0)
#
#
# sampler.fit(X)
# assert sampler.get_centroid_samples() == [7, 16, 29]
# print("close samples assertion passed")
# assert sampler.get_outlier_samples() == [18, 6, 24]
# print("farest samples assertion passed")
# # print(sampler.get_random_samples(5))
# print(sampler.get_samples(n_samples_random=5))

#cluster.get_samples(n_samples=10)
# print(cluster)