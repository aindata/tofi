from abc import ABC, abstractmethod


class AbstractCluster(ABC):
    """
    Abstracts the general logic behind cluster-based sampling
    """

    @abstractmethod
    def get_samples(self, n_samples):
        pass

    @abstractmethod
    def get_centroid_samples(self):
        pass

    @abstractmethod
    def get_outlier_samples(self):
        pass

    @abstractmethod
    def get_random_samples(self, n_samples: int):
        pass
