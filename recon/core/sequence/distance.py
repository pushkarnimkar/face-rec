from .base import BaseSequencer
from .utils import dist_mat_order, make_cluster_estimator

import numpy as np


class DistanceMatrixSequencer(BaseSequencer):
    name = "distance_matrix_order"

    def __init__(self, cluster_method: str= "dbscan"):
        self.cluster_method = cluster_method

    def sequence(self, encs: np.ndarray) -> np.ndarray:
        if encs.shape[0] < 8:
            return np.arange(encs.shape[0])

        cluster_estimator = make_cluster_estimator(self.cluster_method)
        clusters, orders = cluster_estimator.fit_predict(encs), []
        for ci in np.unique(clusters):
            ci_index = np.where(clusters == ci)[0]
            ci_points = encs[ci_index]
            orders.append(ci_index[dist_mat_order(ci_points)])

        _max_size = max(map(lambda x: x.shape[0], orders))
        return np.array([
            orders[ci][idx] for idx in range(_max_size)
            for ci in np.unique(clusters) if idx < orders[ci].shape[0]
        ])
