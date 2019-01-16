"""
Author: Michał Sypetkowski

Implementation of Hierarchical Agglomerative Clustering
(https://en.wikipedia.org/wiki/Hierarchical_clustering)
"""

import numpy as np


class Cluster:
    def __init__(self, point, distances):
        self.points = [point]
        self.id = point[0]
        self.valid = True
        self.distances=distances

    def merge(self, other):
        self.points.extend(other.points)
        other.points = []
        other.valid = False


class MyHAGClustering:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, data):
        """ Returns list of cluster indices.
        """
        distances = np.ones((len(data), len(data))) * float("inf")
        clusters = [Cluster((i, row), distances) for i, row in enumerate(data)]
        for c1 in clusters:
            for c2 in clusters:
                if c1 is not c2:
                    distances[c1.id, c2.id] = np.linalg.norm(c1.points[0][1] - c2.points[0][1])
                    distances[c2.id, c1.id] = distances[c1.id, c2.id]
        distances_sum = distances.copy()
        # print(distances[:5,:3])

        while sum([c.valid for c in clusters]) > self.n_clusters:
            m = np.unravel_index(distances.argmin(), distances.shape)
            # print(m)
            c1 = clusters[m[0]]
            c2 = clusters[m[1]]
            assert(c1.points)
            assert(c2.points)
            for c in clusters:
                if c.valid and c is not c1 and c is not c2:
                    distances_sum[c1.id, c.id] = distances_sum[c1.id, c.id] + distances_sum[c2.id, c.id]
                    distances_sum[c.id, c1.id] = distances_sum[c1.id, c.id]
                    distances[c1.id, c.id] = distances_sum[c1.id, c.id] / len(c1.points) / len(c2.points)
                    distances[c.id, c1.id] = distances[c1.id, c.id]
            for c in clusters:
                if c.valid:
                    distances_sum[c.id, c2.id] = float("inf")
                    distances_sum[c2.id, c.id] = float("inf")
                    distances[c.id, c2.id] = float("inf")
                    distances[c2.id, c.id] = float("inf")
            c1.merge(c2)

        ret = np.zeros(len(data), dtype=int)
        id = 0
        for c in clusters:
            if c.valid:
                for i, _ in c.points:
                    ret[i] = id
                id += 1
        self.labels_ = ret
        # print(ret)
        return ret
