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

from clustering import AbstractCluster


class KMeansCluster(AbstractCluster, ABC):
    # TODO: Return indexes of X that are used in sampling. Therefore, we can use the indexes to get the original data.
    def __init__(self,
                 n_clusters=8,
                 max_features=None,
                 max_iter=1000,
                 **kwargs):
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.X = None
        self.X_pca = None
        self.model = KMeans(n_clusters=n_clusters,
                            max_iter=max_iter,
                            **kwargs)
        self.centroids = None
        self.idx_to_cluster = None  # an array holding cluster_ids for each instance
        self.distances = {}
        self.selected_indexes = {}
        self.close_samples = {}
        self.far_samples = {}
        self.random_samples = {}
        self.close_instances_stacked = None
        self.far_instances_stacked = None
        self.random_instances_stacked = None
        self.all_samples_stacked = None
        for i in range(n_clusters):
            self.selected_indexes[i] = []
            self.close_samples[i] = None
            self.far_samples[i] = None
            self.random_samples[i] = None

    def _fit(self, data):
        if data.shape[0] < self.n_clusters:
            print(f'Number of clusters is greater than number of instances in the dataset.')  # TODO: raise an error
            return None
        self.X = data
        if self.max_features is not None:
            self.X_pca = PCA(n_components=self.max_features).fit_transform(self.X)
        self.model = self.model.fit(self.X_pca if self.X_pca is not None else self.X)
        self.centroids = self.model.cluster_centers_
        self.idx_to_cluster = self.model.labels_

    def get_centroids(self):
        return self.centroids

    def get_labels(self):
        return self.idx_to_cluster

    def calculate_cluster_distances(self, distance_metric='cosine'):
        """
        # TODO: docstring
        :param distance_metric:
        :return:
        """
        for i, c in enumerate(self.centroids):
            one_cluster = self.X[np.where(self.labels == i)]
            centroid_with_group = np.vstack([one_cluster, c])
            distance_matrix = pairwise_distances(centroid_with_group, metric=distance_metric)
            distances = distance_matrix[:-1, -1]
            self.distances[i] = distances

    def get_close_samples(self, n: int, cluster_id: int):  # TODO: computes centroid
        one_cluster = self.X[np.where(self.labels == cluster_id)]
        # Same with argsort but faster.
        min_distance_indexes = np.argpartition(self.distances[cluster_id], n)[:n]
        # min_distance_indexes = np.argsort(self.distances[cluster_id])[:n]  # Same with above.
        min_distance_indexes = [idx for idx in min_distance_indexes if idx not in self.selected_indexes[cluster_id]]
        self.save_used_indexes(min_distance_indexes, cluster_id)
        close_instances = one_cluster[min_distance_indexes]
        return close_instances

    def get_centroid_samples(self):
        """

        :return:
        """
        pass

    def get_far_samples(self, n: int, cluster_id: int):
        one_cluster = self.X[np.where(self.labels == cluster_id)]
        max_distance_indexes = np.argpartition(self.distances[cluster_id], -n)[-n:]
        max_distance_indexes = [idx for idx in max_distance_indexes if idx not in self.selected_indexes[cluster_id]]
        self.save_used_indexes(max_distance_indexes, cluster_id)
        far_instances = one_cluster[max_distance_indexes]
        return far_instances

    def get_random_samples(self, n: int, cluster_id: int):
        one_cluster = self.X[np.where(self.labels == cluster_id)]
        one_cluster = np.delete(one_cluster, self.selected_indexes[cluster_id])
        random_indexes = np.random.choice(len(one_cluster), n, replace=False)
        # TODO: we may get less number of indexes or this can fail.
        self.save_used_indexes(random_indexes, cluster_id)
        random_instances = one_cluster[random_indexes]
        return random_instances

    def save_used_indexes(self, arr: np.array, cluster_id: int):
        self.selected_indexes[cluster_id].extend(arr)

    def stack_results(self):
        close_samples_filled = {k: v for k, v in self.close_samples.items() if v is not None}
        far_samples_filled = {k: v for k, v in self.far_samples.items() if v is not None}
        random_samples_filled = {k: v for k, v in self.random_samples.items() if v is not None}
        stack_list = []
        if close_samples_filled:
            self.close_instances_stacked = np.vstack(close_samples_filled.values())
            np.random.shuffle(self.close_instances_stacked)  # TODO: you do not need to shuffle what to sample
            stack_list.append(self.close_instances_stacked)
        if far_samples_filled:
            self.far_instances_stacked = np.vstack(far_samples_filled.values())
            np.random.shuffle(self.far_instances_stacked)   # TODO: you do not need to shuffle what to sample
            stack_list.append(self.far_instances_stacked)
        if random_samples_filled:
            self.random_instances_stacked = np.vstack(random_samples_filled.values())
            np.random.shuffle(self.random_instances_stacked)  # TODO: you do not need to shuffle what to sample
            stack_list.append(self.random_instances_stacked)
        self.all_samples_stacked = np.concatenate(stack_list)
        return self.all_samples_stacked

    def _get_n_samples_per_cluster(self, n_samples: int):
        # Calculate number of samples for each cluster
        n_samples_per_cluster = n_samples // self.n_clusters
        cluster_remainder = n_samples % self.n_clusters
        # TODO: int shouldn't change to array
        n_samples_per_cluster = np.array([n_samples_per_cluster] * self.n_clusters)
        while cluster_remainder > 0:
            n_samples_per_cluster[np.argmin(n_samples_per_cluster)] += 1
            cluster_remainder -= 1
        return n_samples_per_cluster

    def get_samples(self, n_samples: int, distance_metric='cosine'):
        """
        Algorithm:
        1. compute cluster_centroid_samples -> this assumed to be equal to number of centroids
        2. compute outlier_samples -> far samples from the cluster centroids
        3. compute random_samples -> randomly choose samples that are unselected in previous steps

        :param n_samples: total number of samples needed
        :param distance_metric: distance function used to compute the distances
        :return:
        """

        if n_samples > self.X.shape[0]:
            print(f'Number of samples is less than number of instances in the dataset.'
                  f'Please, decrease n value to {self.X.shape[0]}.')
            return None  # TODO: raise an error
        elif n_samples == self.X.shape[0]:
            return self.X
        if n_samples < self.n_clusters:
            print(f'Number of samples is less than number of clusters.'
                  f'Please, decrease n value to {self.n_clusters}.')
            return None  # TODO: raise an error
        # Calculate distances between centroids and instances.
        self.calculate_cluster_distances(distance_metric=distance_metric)
        n_samples_per_cluster = self._get_n_samples_per_cluster(n_samples)
        # For every cluster, calculate number of samples for each type of sample and collect
        for i in range(self.n_clusters):
            if n_samples_per_cluster[i] == 1:
                self.close_samples[i] = self.get_close_samples(1, i)
            elif n_samples_per_cluster[i] == 2:
                self.close_samples[i] = self.get_close_samples(1, i)
                self.far_samples[i] = self.get_far_samples(1, i)
            else:
                close_sample_size = np.ceil(n_samples * 0.2).astype(int)
                far_sample_size = np.ceil(n_samples * 0.2).astype(int)
                random_sample_size = n_samples - (close_sample_size + far_sample_size)
                self.close_samples[i] = self.get_close_samples(close_sample_size, i)
                self.far_samples[i] = self.get_far_samples(far_sample_size, i)
                self.random_samples[i] = self.get_random_samples(random_sample_size, i)
        self.stack_results()


# EXAMPLE OF USAGE
X, y = make_blobs(n_samples=30, n_features=2)
print(X.shape)
cluster = KMeansCluster(n_clusters=3)

cluster._fit(X)
cluster.get_samples(n_samples=10)
print(cluster)
