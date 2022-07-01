from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.datasets import make_blobs


# TODO: Return indexes of X that are used in sampling. Therefore, we can use the indexes to get the original data.

class Cluster:
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
        self.labels = None
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

    def fit(self, data):
        if data.shape[0] < self.n_clusters:
            print(f'Number of clusters is greater than number of instances in the dataset.')
            return None
        self.X = data
        if self.max_features is not None:
            self.X_pca = PCA(n_components=self.max_features).fit_transform(self.X)
        self.model = self.model.fit(self.X_pca if self.X_pca is not None else self.X)
        self.get_centroids()
        self.get_labels()
        return self.model

    def get_centroids(self):
        self.centroids = self.model.cluster_centers_
        return self.centroids

    def get_labels(self):
        self.labels = self.model.labels_
        return self.labels

    def calculate_cluster_distances(self, distance_metric='cosine'):
        for i, c in enumerate(self.centroids):
            one_cluster = self.X[np.where(self.labels == i)]
            centroid_with_group = np.vstack([one_cluster, c])
            distance_matrix = pairwise_distances(centroid_with_group, metric=distance_metric)
            distances = distance_matrix[:-1, -1]
            self.distances[i] = distances

    def get_close_samples(self, n: int, cluster_id: int):
        one_cluster = self.X[np.where(self.labels == cluster_id)]
        # Same with argsort but faster.
        min_distance_indexes = np.argpartition(self.distances[cluster_id], n)[:n]
        # min_distance_indexes = np.argsort(self.distances[cluster_id])[:n]  # Same with above.
        min_distance_indexes = [idx for idx in min_distance_indexes if idx not in self.selected_indexes[cluster_id]]
        self.save_used_indexes(min_distance_indexes, cluster_id)
        close_instances = one_cluster[min_distance_indexes]
        return close_instances

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
            np.random.shuffle(self.close_instances_stacked)
            stack_list.append(self.close_instances_stacked)
        if far_samples_filled:
            self.far_instances_stacked = np.vstack(far_samples_filled.values())
            np.random.shuffle(self.far_instances_stacked)
            stack_list.append(self.far_instances_stacked)
        if random_samples_filled:
            self.random_instances_stacked = np.vstack(random_samples_filled.values())
            np.random.shuffle(self.random_instances_stacked)
            stack_list.append(self.random_instances_stacked)
        self.all_samples_stacked = np.concatenate(stack_list)
        return self.all_samples_stacked

    def get_samples(self, n: int, distance_metric='cosine'):
        if n > self.X.shape[0]:
            print(f'Number of samples is less than number of instances in the dataset.'
                  f'Please, decrease n value to {self.X.shape[0]}.')
            return None
        elif n == self.X.shape[0]:
            return self.X
        if n < self.n_clusters:
            print(f'Number of samples is less than number of clusters.'
                  f'Please, decrease n value to {self.n_clusters}.')
            return None
        # Calculate distances between centroids and instances.
        self.calculate_cluster_distances(distance_metric=distance_metric)
        # Calculate number of samples for each cluster
        n_samples_per_cluster = n // self.n_clusters
        cluster_remainder = n % self.n_clusters
        n_samples_per_cluster = np.array([n_samples_per_cluster] * self.n_clusters)
        while cluster_remainder > 0:
            n_samples_per_cluster[np.argmin(n_samples_per_cluster)] += 1
            cluster_remainder -= 1
        # For every cluster, calculate number of samples for each type of sample and collect
        for i in range(self.n_clusters):
            if n_samples_per_cluster[i] == 1:
                self.close_samples[i] = self.get_close_samples(1, i)
            elif n_samples_per_cluster[i] == 2:
                self.close_samples[i] = self.get_close_samples(1, i)
                self.far_samples[i] = self.get_far_samples(1, i)
            else:
                close_sample_size = np.ceil(n * 0.2).astype(int)
                far_sample_size = np.ceil(n * 0.2).astype(int)
                random_sample_size = n - (close_sample_size + far_sample_size)
                self.close_samples[i] = self.get_close_samples(close_sample_size, i)
                self.far_samples[i] = self.get_far_samples(far_sample_size, i)
                self.random_samples[i] = self.get_random_samples(random_sample_size, i)
        self.stack_results()


# EXAMPLE OF USAGE
X, y = make_blobs(n_samples=30, n_features=2)
print(X.shape)
cluster = Cluster(n_clusters=3)
cluster.fit(X)
cluster.get_samples(n=10)
print(cluster)
