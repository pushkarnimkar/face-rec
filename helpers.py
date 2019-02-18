from typing import Tuple, List

import numpy as np


def shuffle(arrs: List[np.ndarray]) -> List[np.ndarray]:
    if len(arrs) > 0:
        order = np.random.permutation(arrs[0].shape[0])
        return [np.take(arr, order, axis=0) for arr in arrs]
    return arrs


def make_bare_train_split(x: np.ndarray, y: np.ndarray, min_count=3) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    order = np.argsort(y)
    x_sorted, y_sorted = x[order, :], y[order]
    _, indices, counts = np.unique(y_sorted, return_index=True,
                                   return_counts=True)

    min_count = np.min(counts) if np.min(counts) < min_count else min_count
    bare_train_order_index = np.ndarray((indices.shape[0], 0), dtype=np.int)
    for inc in range(min_count):
        bare_train_order_index = np.hstack((bare_train_order_index,
                                            indices.reshape(-1, 1) + inc))

    bare_train_index = bare_train_order_index.flatten()
    rest_index = np.setdiff1d(np.arange(x.shape[0]), bare_train_index)
    x_bare_train, y_bare_train = \
        x_sorted[bare_train_index, :], y_sorted[bare_train_index]
    x_rest, y_rest = x_sorted[rest_index, :], y_sorted[rest_index]

    return x_bare_train, y_bare_train, x_rest, y_rest


def train_test_split(x: np.ndarray, y: np.ndarray, train_fract: float=0.6,
                     train_min_count: int=3) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    x_bare_train, y_bare_train, x_rest, y_rest = \
        make_bare_train_split(x, y, min_count=train_min_count)
    train_count, bare_train_count = (
        int(train_fract * x.shape[0]), x_bare_train.shape[0])

    assert train_count >= x_bare_train.shape[0]
    rest_train_count = train_count - x_bare_train.shape[0]

    perm = np.random.permutation(x_rest.shape[0])
    x_rest_train, y_rest_train = (
        x_rest[perm[:rest_train_count]], y_rest[perm[:rest_train_count]])
    x_test, y_test = (
        x_rest[perm[rest_train_count:]], y_rest[perm[rest_train_count:]])

    x_train, y_train = shuffle([np.vstack((x_rest_train, x_bare_train)),
                                np.concatenate((y_rest_train, y_bare_train))])
    return x_train, y_train, x_test, y_test
