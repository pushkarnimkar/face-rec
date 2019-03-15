from .base import BaseSequencer
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from typing import Optional
from .utils import dist_mat_order, make_transformer, make_cluster_estimator

import numpy as np
import sys


def select_hull(clusters, transformed) -> np.ndarray:
    points, weights, indices, nans_index, selected = [], [], [], [], []

    for cluster_index in np.unique(clusters):
        cluster_points_index = np.where(clusters == cluster_index)[0]
        cluster_points = transformed[cluster_points_index]

        if cluster_index != -1 and cluster_points.shape[0] > 2:
            try:
                hull = ConvexHull(cluster_points)
                size = hull.vertices.shape[0]
                values = (np.arange(0, 1, 1 / size) +
                          np.random.normal(0, 1 / size / 5, size))
                order = dist_mat_order(hull.points[hull.vertices])

                weights.append(values[order])
                indices.append(cluster_points_index[hull.vertices])
            except QhullError as qhe:
                print(qhe, file=sys.stderr)
                nans_index.append(cluster_points_index)
        else:
            nans_index.append(cluster_points_index)

    if len(nans_index) != 0:
        nans_index = np.concatenate(nans_index)
        size = nans_index.shape[0]
        # print("nans_index.shape  ->", nans_index.shape)
        weights.append(((np.arange(0, 1, 1 / size) +
                         np.random.normal(0, 1 / size / 5)) / 4)
                       [np.random.permutation(size)])
        indices.append(nans_index)

    weights, indices = np.concatenate(weights), np.concatenate(indices)
    return indices[weights.argsort()]


class ConvexHullSequencer(BaseSequencer):
    _name_ = "ConvexHullSequencer"

    def __init__(self, cluster_method: Optional[str]="dbscan",
                 transform_method: Optional[str]="tsne",
                 minimum_sequence: Optional[int]=8, **kwargs):

        self.cluster_method = cluster_method
        self.transform_method = transform_method
        self.minimum_sequence = minimum_sequence
        self.kws = kwargs

    def sequence(self, encs: np.ndarray) -> np.ndarray:
        if encs.shape[0] < self.minimum_sequence:
            return np.arange(encs.shape[0])

        cluster_estimator = make_cluster_estimator("dbscan", **self.kws)
        clusters = cluster_estimator.fit_predict(encs)
        transformer = make_transformer("tsne", **self.kws)
        transformed = transformer.fit_transform(encs)

        selected = select_hull(clusters, transformed)
        extras = set(range(encs.shape[0])).difference(selected)
        return np.concatenate((selected, np.array(list(extras))))


class IterativeHullSequencer(BaseSequencer):
    _name_ = "IterativeHullSequencer"

    def __init__(self, minimum_sequence: Optional[int]=8):
        self.minimum_sequence = minimum_sequence

    def sequence(self, encs: np.ndarray) -> np.ndarray:
        selected = np.ndarray((0,), dtype=np.int)
        available, available_index = encs, np.arange(encs.shape[0])
        while available.shape[0] != 0:
            if available.shape[0] < self.minimum_sequence:
                selected_now = np.arange(available.shape[0])
            else:
                cluster_estimator = make_cluster_estimator("dbscan")
                clusters = cluster_estimator.fit_predict(available)
                transformer = make_transformer("tsne")
                transformed = transformer.fit_transform(available)
                selected_now = select_hull(clusters, transformed)

            selected_glob = available_index[selected_now]
            selected = np.concatenate((selected, selected_glob))

            difference = np.setdiff1d(np.arange(available.shape[0]), selected_now)
            available = available[difference, :]
            available_index = available_index[difference]

        return selected
