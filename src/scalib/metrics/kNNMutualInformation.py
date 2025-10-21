import numpy as np

from scipy.special import digamma, gamma
from scipy.spatial import cKDTree


class kNNInformationEstimator:
    r"""Mutual Information Estimator for discrete X and continous Y

    Based on "Mutual Information between Discrete and Continuous Data Sets" from Brian C. Ross
    """

    def __init__(self, M: int, X: np.ndarray[np.uint32], Y: np.ndarray[float]):

        self.Nsample = len(X)
        self.M = M
        self.sample_per_class = np.zeros(M, dtype=np.uint32)

        self.main_tree = cKDTree(Y, leafsize=8)
        self.list_class_trees = []
        for i in range(self.M):
            self.sample_per_class[i] = np.sum(X == i)
            self.list_class_trees.append(cKDTree(Y[X == i], leafsize=8))

        self.max_k = np.min(self.sample_per_class) - 1

    def predict(self, k, p=2, base=2):

        if k > self.max_k:
            raise ValueError(
                f"Invalid Inputs, with these samples k can be at most {self.max_k} which is less than k = {k}"
            )

        digamma_neigh = 0
        for num_class in range(self.M):
            d, _ = self.list_class_trees[num_class].query(
                self.list_class_trees[num_class].data, k + 1, p=p, workers=1, eps=0
            )

            # !!!! The center of the ball should not be counted !!!!
            num_neigh = (
                self.main_tree.query_ball_point(
                    self.list_class_trees[num_class].data,
                    d[:, k],
                    return_length=True,
                    workers=1,
                    eps=0,
                )
                - 1
            )
            digamma_neigh += np.sum(digamma(num_neigh))
        digamma_neigh /= self.Nsample

        mi_kNN = (
            digamma(self.Nsample)
            + digamma(k)
            - np.sum(self.sample_per_class * digamma(self.sample_per_class))
            / self.Nsample
            - digamma_neigh
        )
        mi_kNN /= np.log(base)

        return np.maximum(0, mi_kNN)

    def ensemble_predict(self, ensemble_k, p=2, base=2):
        ensemble_k = np.asarray(ensemble_k)
        max_k_ensemble = np.max(ensemble_k)

        if max_k_ensemble >= self.max_k:
            raise ValueError(
                f"Invalid Inputs, with these samples k can be at most {self.max_k} which is less than k = {max_k_ensemble}"
            )

        digamma_neigh = np.zeros(len(ensemble_k))
        for num_class in range(self.M):
            d, _ = self.list_class_trees[num_class].query(
                self.list_class_trees[num_class].data,
                max_k_ensemble + 1,
                p=p,
                workers=1,
                eps=0,
            )

            for i in range(len(ensemble_k)):
                # !!!! The center of the ball should not be counted !!!!
                k = ensemble_k[i]
                num_neigh = (
                    self.main_tree.query_ball_point(
                        self.list_class_trees[num_class].data,
                        d[:, k],
                        return_length=True,
                        workers=1,
                        eps=0,
                    )
                    - 1
                )
                digamma_neigh[i] += np.sum(digamma(num_neigh))
        digamma_neigh /= self.Nsample

        mi_kNNs = (
            digamma(self.Nsample)
            + digamma(ensemble_k)
            - np.sum(self.sample_per_class * digamma(self.sample_per_class))
            / self.Nsample
            - digamma_neigh
        )
        mi_kNNS = mi_kNNs / np.log(base)
        mi_kNNS = np.maximum(0, mi_kNNS)

        # Taking maximum inside the mean increases the MSE but deacreases the risk to underestimate the Mutual Information
        # Otherwise we can take the median ?
        return np.median(mi_kNNS), mi_kNNS
