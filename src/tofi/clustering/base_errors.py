class NotEnoughPointsError(Exception):
    """
    Exception raised if there are more
    clusters than the data points
    """
    def __init__(self,
                 data_size=None,
                 n_clusters=None,
                 message='Number of clusters is greater than number of instances in the dataset.'):
        self.n_data = data_size
        self.n_clusters = n_clusters
        super(NotEnoughPointsError, self).__init__()

    def __str__(self):
        return f"{self.n_data} can't be less than {self.n_clusters}"


class ClusterDataAlreadySet(Exception):
    """
    Exception raised if the data is already set for current Sampler
    """
    def __init__(self, 
                 message='You already set the data for this sampler'):
        super(ClusterDataAlreadySet, self).__init__()
        


