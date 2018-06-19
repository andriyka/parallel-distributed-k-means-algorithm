import itertools
import random
from multiprocessing import Process, Array, Manager
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
from collections import OrderedDict

import numpy as np

manager = Manager()


class KMeansAlgorithm(object):
    def __init__(self, num_clusters, max_iter=1000, eps=0.001):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.eps = eps
        self.initial_centroids = None
        np.random.seed(152)

    def fit(self, X, par = False):
        self.initial_centroids = self.initial_centroids if self.initial_centroids is not None else self._init_centroids(X)
        centroids = np.asarray(self.initial_centroids)
        objective_history = []
        convergence = False
        iteration = 0
        while not convergence:
            closest_centroids = self.get_closest_centroids_par(X, centroids) if par else \
                self._get_closest_centroids(X, centroids)
            centroids = self._move_centroids(X, closest_centroids)
            objective = self._kmeans_objective(X, centroids, closest_centroids)
            objective_history.append(objective)
            iteration += 1
            convergence = len(objective_history) > 2 and (
                    objective_history[-2] / objective_history[-1] < 1.01 or iteration > self.max_iter)
            print("Iteration: {0:2d}    Objective: {1:.3f}".format(iteration, objective))
        return objective_history

    def _init_centroids(self, X):
        return np.asarray([X[i] for i in np.random.choice(range(0, len(X)), self.num_clusters, replace=False)])

    @staticmethod
    def _get_closest_centroids(X, centroids):
        res = [np.argmin([np.sum(np.power(x - c, 2)) for c in centroids]) for x in X]
        return np.asarray(res)

    @staticmethod
    def get_closest_centroids_par(X, centroids):
        """
        Parallel MapReduce K-means; Map & Reduce could be separated into two functions but it is not
        done for performance.
        :param X: training set
        :param centroids: current centroids
        :return: array of size X with centroid number for each x
        """
        def get_centroids_local(chunk, rr, i):
            rr[i] = [np.argmin([np.sum(np.power(x - c, 2)) for c in centroids]) for x in chunk]

        results = manager.dict()
        chunks = np.split(X, 16)

        processes = []

        # Map
        for i, chunk in enumerate(chunks):
            t = mp.Process(target=get_centroids_local, args=(chunk, results, i))
            t.start()
            processes.append(t)

        for t in processes:
            t.join()

        # Reduce
        res = []
        for k in sorted(results.keys()):
            res.extend(results[k])

        return np.asarray(res)

    def _move_centroids(self, X, closest_centroids):
        res = np.zeros((self.num_clusters, X.shape[-1]))
        for i in range(self.num_clusters):
            assigned_points = X[closest_centroids == i]
            res[i] = np.mean(assigned_points, axis=0)
        return res

    @staticmethod
    def _kmeans_objective(X, centroids, closest_centroids):
        cost_sum = 0
        for i in range(len(centroids)):
            assigned_points = X[closest_centroids == i]
            cost_sum += np.sum(np.power(assigned_points - centroids[i], 2))

        return cost_sum


def perform_test(input_size=2**10):
    alg = KMeansAlgorithm(8)


    times_s = []
    times_p = []

    for i in range(5, 14):
        X = np.random.rand(2**i, 3)
        ts = time.time()
        res = alg.fit(X)
        times_s.append(time.time() - ts)

        tp = time.time()
        res1 = alg.fit(X, par=True)
        times_p.append((time.time() - tp))

    plt.plot([2**i for i in range(5,14)], times_s)
    plt.plot([2**i for i in range(5,14)], times_p)

    plt.legend(['seq', 'par'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    perform_test()




