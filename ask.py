from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.qhull import QhullError
# from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

import numpy as np
import sys


def cluster(encs: np.ndarray, n_clusters=8, init="k-means++") -> np.ndarray:
    dbscan = DBSCAN(eps=0.425, min_samples=2)
    return dbscan.fit_predict(encs)


def transform(encs: np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=2, init="pca", random_state=100, method="exact")
    return tsne.fit_transform(encs)


def make_mask(side, index) -> np.ndarray:
    square = np.repeat(True, side * side).reshape(side, side)
    square[:, index] = False
    square[index, :] = False
    return square


def excl_from_dist_mat(dist_mat, index) -> np.ndarray:
    assert dist_mat.shape[0] > 1, "too small matrix"
    assert dist_mat.shape[0] == dist_mat.shape[1], "malformed distance matrix"

    side = dist_mat.shape[0]
    mask = make_mask(side, index)

    removed = 1 if np.isscalar(index) else len(index)
    return (dist_mat.flatten()[mask.flatten()]
            .reshape(side - removed, side - removed))


def dist_mat_order(encs) -> np.ndarray:
    dist_mat = distance_matrix(encs, encs)
    selection = list(np.unravel_index(dist_mat.argmax(), dist_mat.shape))
    available, available_mat = (np.delete(np.arange(dist_mat.shape[0]), selection),
                                excl_from_dist_mat(dist_mat, selection))
    while available.shape[0] != 0:
        next_point = np.argmax(dist_mat[available].sum(1))
        selection.append(available[next_point])
        if available.shape[0] == 1:
            available = np.delete(available, next_point)
        else:
            available, available_mat = (np.delete(available, next_point),
                                        excl_from_dist_mat(dist_mat, selection))
    return np.array(selection)


def select(clusters, transformed) -> np.ndarray:
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


def convex_hull_sequence(encs: np.ndarray, kmeans_n_clusters=8) -> np.ndarray:
    if encs.shape[0] < kmeans_n_clusters:
        return np.arange(encs.shape[0])

    clusters, transformed = cluster(encs), transform(encs)
    selected = select(clusters, transformed)
    extras = set(range(encs.shape[0])).difference(selected)
    return np.concatenate((selected, np.array(list(extras))))


def random_sequence(encs: np.ndarray) -> np.ndarray:
    return np.random.permutation(encs.shape[0])


def iterative_hull_sequence(encs: np.ndarray) -> np.ndarray:
    min_points = 8

    selected = np.ndarray((0,), dtype=np.int)
    available, available_index = encs, np.arange(encs.shape[0])
    while available.shape[0] != 0:
        if available.shape[0] < min_points:
            selected_now = np.arange(available.shape[0])
        else:
            clusters, transformed = cluster(available), transform(available)
            selected_now = select(clusters, transformed)

        selected_glob = available_index[selected_now]
        selected = np.concatenate((selected, selected_glob))

        difference = np.setdiff1d(np.arange(available.shape[0]), selected_now)
        available = available[difference, :]
        available_index = available_index[difference]

    return selected


def dist_mat_order_sequence(encs: np.ndarray) -> np.ndarray:
    if encs.shape[0] < 8:
        return np.arange(encs.shape[0])

    clusters, cluster_orders = cluster(encs), []
    for cluster_index in np.unique(clusters):
        cluster_points_index = np.where(clusters == cluster_index)[0]
        cluster_points = encs[cluster_points_index]
        order = dist_mat_order(cluster_points)
        cluster_orders.append(cluster_points_index[order])

    max_cluster_size, final_order = \
        max(map(lambda x: x.shape[0], cluster_orders)), []

    final_order = [cluster_orders[cluster_index][index]
                   for index in range(max_cluster_size)
                   for cluster_index in np.unique(clusters)
                   if index < cluster_orders[cluster_index].shape[0]]

    return np.array(final_order)
