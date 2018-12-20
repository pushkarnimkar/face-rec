from scipy.spatial import ConvexHull
# from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

import numpy as np


def cluster(encs: np.ndarray, n_clusters=8, init="k-means++") -> np.ndarray:
    dbscan = DBSCAN(eps=0.425, min_samples=2)
    return dbscan.fit_predict(encs)


def transform(encs: np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=2, init="pca", random_state=100, method="exact")
    return tsne.fit_transform(encs)


def select(clusters, transformed):
    (points, weights, indices,
     nans, nans_index, selected) = ([], [], [], [], [], [])

    for cluster_idx in np.unique(clusters):
        cluster_points_index = np.where(clusters == cluster_idx)[0]
        cluster_points = transformed[cluster_points_index]

        if cluster_idx != -1 and cluster_points.shape[0] > 2:
            hull = ConvexHull(cluster_points)
            size = hull.vertices.shape[0]

            weights.append((np.arange(0, 1, 1 / size) +
                            np.random.normal(0, 1 / size / 5, size))
                           [np.random.permutation(size)])
            indices.append(cluster_points_index[hull.vertices])
        else:
            nans_index.append(cluster_points_index)

    if len(nans) != 0:
        nans_index = np.concatenate(nans_index)
        size = nans_index.shape[0]
        print("nans_index.shape  ->", nans_index.shape)
        weights.append(((np.arange(0, 1, 1 / size) +
                         np.random.normal(0, 1 / size / 5)) / 4)
                       [np.random.permutation(size)])
        indices.append(nans_index)

    weights, indices = np.concatenate(weights), np.concatenate(indices)
    return indices[weights.argsort()]


def convex_hull_sequence(encs: np.ndarray, kmeans_n_clusters=8) -> np.ndarray:
    if encs.shape[0] < kmeans_n_clusters:
        return np.arange(0, encs.shape[0])

    clusters, transformed = cluster(encs), transform(encs)
    selected = select(clusters, transformed)
    extras = set(range(encs.shape[0])).difference(selected)
    return np.concatenate((selected, np.array(list(extras))))


def random_sequence(encs: np.ndarray) -> np.ndarray:
    return np.random.permutation(encs.shape[0])
