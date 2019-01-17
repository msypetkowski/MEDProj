"""
Author: Michał Sypetkowski

Implementation of Hierarchical Agglomerative Clustering
(https://en.wikipedia.org/wiki/Hierarchical_clustering)

Implemented is 2 variants:

One with time complexity of O(n^3), and memory O(n^2).
The other version has time complexity O(n^2*log(n))
It optimizes minimum index finding with priority queue.
Here, this operation (np.unrevel_index) has anyway relatively low time coefficient,
because the method is implemented in C++ in numpy, and python (interpreter) overhead is relatively high.
"""

import numpy as np


class Cluster:
    def __init__(self, point):
        self.points = [point]
        self.id = point[0]
        self.valid = True

    def merge(self, other):
        self.points.extend(other.points)
        other.points = []
        other.valid = False

    def __lt__(self, other):
        return self.id < other.id


class MyHAGClustering:

    def __init__(self, n_clusters, use_heap=True):
        self.n_clusters = n_clusters
        self.use_heap = use_heap

    def fit_On3(self, data):
        """ Returns list of cluster indices.
        O(n^3) implementation.
        """
        distances = np.ones((len(data), len(data))) * float("inf")
        clusters = [Cluster((i, row)) for i, row in enumerate(data)]
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

    def fit_On2logn(self, data):
        """ Returns list of cluster indices.
        O(n^2*log(n)) implementation.
        """
        import heapq

        heap = []

        distances_sum = np.ones((len(data), len(data))) * float("inf")
        clusters = [Cluster((i, row)) for i, row in enumerate(data)]
        for c1 in clusters:
            for c2 in clusters:
                if c1.id < c2.id:
                    distances_sum[c1.id, c2.id] = np.linalg.norm(c1.points[0][1] - c2.points[0][1])
                    heapq.heappush(heap, (distances_sum[c1.id, c2.id], c1, c2, len(c1.points), len(c2.points)))

        while sum([c.valid for c in clusters]) > self.n_clusters:
            while True:
                d, c1, c2, l1, l2 = heapq.heappop(heap)
                if c1.valid and c2.valid and l1 == len(c1.points) and l2 == len(c2.points):
                    break
            assert(c1.points)
            assert(c2.points)
            count1, count2 = len(c1.points), len(c2.points)
            c1.merge(c2)
            for c in clusters:
                if c.valid and c is not c1 and c is not c2:
                    clu = [c1, c]
                    i1, i2 = min(clu), max(clu)
                    distances_sum[i1.id, i2.id] = distances_sum[i1.id, i2.id] \
                            + distances_sum[min(c2.id,c.id), max(c2.id, c.id)]
                    heapq.heappush(heap, (distances_sum[i1.id, i2.id]
                        / count1 / count2, i1, i2, len(i1.points), len(i2.points)))

        ret = np.zeros(len(data), dtype=int)
        id = 0
        for c in clusters:
            if c.valid:
                for i, _ in c.points:
                    ret[i] = id
                id += 1
        self.labels_ = ret
        return ret

        pass

    def fit(self, data):
        if len(data) < self.n_clusters:
            raise ValueError("Number of clusters is larger than number of samples.")
        if self.use_heap:
            return self.fit_On2logn(data)
        else:
            return self.fit_On3(data)
