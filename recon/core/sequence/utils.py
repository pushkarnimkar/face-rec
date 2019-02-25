from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

import numpy as np


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


def dist_mat_order(encs: np.ndarray) -> np.ndarray:
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


def make_cluster_estimator(method: str, **kwargs):
    if method == "dbscan":
        _eps = kwargs["eps"] if "eps" in kwargs else 0.425
        _min_samples = kwargs["min_samples"] if "min_samples" in kwargs else 2
        return DBSCAN(eps=_eps, min_samples=_min_samples)
    raise ValueError(f"{method} not understood")


def make_transformer(method: str, **kwargs):
    if method == "tsne":
        _init = kwargs["init"] if "init" in kwargs else "pca"
        _n_components = \
            kwargs["n_components"] if "n_components" in kwargs else 2
        _random_state = \
            kwargs["random_state"] if "random_state" in kwargs else 100
        _method = kwargs["method"] if "method" in kwargs else "exact"
        return TSNE(n_components=_n_components, init=_init,
                    random_state=_random_state, method=_method)
    raise ValueError(f"{method} not understood")
