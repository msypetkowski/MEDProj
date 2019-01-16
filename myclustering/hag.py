"""
Author: Michał Sypetkowski

Implementation of Hierarchical Agglomerative Clustering
(https://en.wikipedia.org/wiki/Hierarchical_clustering)

Implemented is variant with time complexity of O(n^3), and memory O(n^2).
O(n^2*log(n)) version optimize minimum index finding with priority queue.
Here, hhis operation has relatively low time coefficient, because the method is
implemented in C++ in numpy. Moreover, it would increase memory coefficient.
Therefore O(n^2*log(n)) implementation may not improve the performance
very significantly in this implementation -- python (interpreter) overhead is high.
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
                if c1.id < c2.id:
                    distances[c1.id, c2.id] = np.linalg.norm(c1.points[0][1] - c2.points[0][1])
        distances_sum = distances.copy()

        while sum([c.valid for c in clusters]) > self.n_clusters:
            m = np.unravel_index(distances.argmin(), distances.shape)
            c1 = clusters[m[0]]
            c2 = clusters[m[1]]
            assert(c1.points)
            assert(c2.points)
            for c in clusters:
                if c.valid and c is not c1 and c is not c2:
                    ids = [c1.id, c.id]
                    i1, i2 = min(ids), max(ids)
                    distances_sum[i1, i2] = distances_sum[i1, i2] + distances_sum[min(c2.id,c.id), max(c2.id, c.id)]
                    distances[i1, i2] = distances_sum[i1, i2] / len(c1.points) / len(c2.points)
            for c in clusters:
                if c.valid:
                    ids = [c.id, c2.id]
                    i1, i2 = min(ids), max(ids)
                    distances[i1, i2] = float("inf")
            c1.merge(c2)

        ret = np.zeros(len(data), dtype=int)
        id = 0
        for c in clusters:
            if c.valid:
                for i, _ in c.points:
                    ret[i] = id
                id += 1
        self.labels_ = ret
        return ret
