"""
Author: Michał Sypetkowski

Implementation of k-means algorithm
(https://en.wikipedia.org/wiki/K-means_clustering).
"""
import numpy as np


class MyKMeans:
    def __init__(self, n_clusters, max_iter=300, initialization_type='MeanStd'):
        """
        Parameters
        ----------
        n_clusters : int
            how much clusters the algorithm will create with fit method

        max_iter : int
            how much assignment and update steps will be repeated at most

        initialization_type : str
            Initial k-means vector inicjalization method.
            Possible values: "MeanStd", "Forgy"
        """
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.initialization_type = initialization_type
        self.rand = np.random.RandomState()

    def fit(self, data):
        """ Returns list of cluster indices.
        """
        if self.initialization_type == 'Forgy':
            kmeans = data[self.rand.choice(data.shape[0], self.n_clusters, replace=False)]
        elif self.initialization_type == 'MeanStd':
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            kmeans = np.random.randn(self.n_clusters, data.shape[1]) * std + mean
        else:
            raise ValueError(f"Unknown initialization_type value: {self.initialization_type}")

        distances = np.zeros((data.shape[0], self.n_clusters))
        for _ in range(self.max_iter):
            for i in range(self.n_clusters):
                distances[:,i] = np.linalg.norm(data - kmeans[i], axis=1)
            labels = np.argmin(distances, axis=1)
            kmeans_old = kmeans.copy()
            for i in range(self.n_clusters):
                mask = labels == i
                kmeans[i] = np.mean(data[mask], axis=0) if mask.any() else kmeans_old[i]
            if (kmeans == kmeans_old).all():
                break

        self.labels_ = labels
        return labels
