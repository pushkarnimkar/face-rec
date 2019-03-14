from .base import BaseSequencer
from .utils import dist_mat_order, make_cluster_estimator

import numpy as np


class DistanceMatrixSequencer(BaseSequencer):
    _name_ = "DistanceMatrixSequencer"

    def __init__(self, cluster_method: str= "dbscan", weight: float=1.0):
        self.cluster_method = cluster_method
        self.weight = weight

    def sequence(self, encs: np.ndarray) -> np.ndarray:
        if encs.shape[0] < 8:
            return np.arange(encs.shape[0])

        cluster_estimator = \
            make_cluster_estimator(self.cluster_method, min_samples=2)
        clusters, orders = cluster_estimator.fit_predict(encs), []
        for ci in np.unique(clusters):
            ci_index = np.where(clusters == ci)[0]
            ci_points = encs[ci_index]
            ci_order = dist_mat_order(ci_points, weight=self.weight)
            orders.append(ci_index[ci_order])

        _max_size = max(map(lambda x: x.shape[0], orders))
        return np.array([
            orders[ci][idx] for idx in range(_max_size)
            for ci in np.unique(clusters) if idx < orders[ci].shape[0]
        ])

    @property
    def name(self):
        return (super(DistanceMatrixSequencer, self).name +
                f"(weight={self.weight})")
