from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

import numpy as np


def make_mask(side, index) -> np.ndarray:
    square = np.repeat(True, side * side).reshape(side, side)
    square[:, index] = False
    square[index, :] = False
    return square


def _next_point(dist_mat: np.ndarray, available: np.ndarray,
                unavailable: list, weight: float):
    """Returns index of next point in available array"""
    comp1 = dist_mat[np.ix_(available, available)].sum(1)
    comp2 = dist_mat[np.ix_(available, unavailable)].sum(1)
    return np.argmax(comp1 + weight * comp2)


def dist_mat_order(encs: np.ndarray, weight: float=1.0) -> np.ndarray:
    """
    Orders encodings as per descending order of distance with respect to
    other points in `encs`. Other points may or may not be already part
    of the sequence.

    Parameters
    ----------
    encs : np.ndarray, n x 128
        Encodings to be sequenced. Contains n 128 dimensional face encodings
    weight : float, optional (default 1.0)
        Weight of second component of distances, that is the multiplication
        factor for distances with included points

    Returns
    -------
    order : np.ndarray, n x 1
        Sequence in which to pick up values from encodings for efficient
        active learning
    """
    dist_mat = distance_matrix(encs, encs)
    available, unavailable = np.arange(dist_mat.shape[0]), []
    order = list(np.unravel_index(dist_mat.argmax(), dist_mat.shape))
    available = np.delete(available, order)
    unavailable += order
    if available.shape[0] == 0:
        return np.array(order)
    while available.shape[0] > 1:
        next_point = _next_point(dist_mat, available, unavailable, weight)
        order.append(available[next_point])
        unavailable.append(available[next_point])
        available = np.delete(available, next_point)
    else:
        order.append(available[0])
    return np.array(order)


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
