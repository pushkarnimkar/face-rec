from recon.core.sequence import dist_mat_order
import numpy as np


def simple_split(x: np.ndarray, y: np.ndarray, train_fract: float=0.6):
    take_train = int(x.shape[0] * train_fract)
    prerm = np.random.permutation(x.shape[0])
    x_train, x_test = x[prerm[:take_train], :], x[prerm[take_train:], :]
    y_train, y_test = y[prerm[:take_train]], y[prerm[take_train:]]
    return x_train, x_test, y_train, y_test


def dmotli_split(x: np.ndarray, y: np.ndarray, train_fract: float=0.6):
    """
    Distance Matrix Ordered - Training Less Information

    Parameters
    ----------
    x : np.ndarray
        Face encodings to be split into two sets
    y : np.ndarray
        Face encoding labels to be split into two sets in
        correspondence with x
    train_fract : float, default = 0.6
        Fractional size of `x` to be used for training

    Returns
    -------
    x_train : np.ndarray, n_train x 128
        Face encodings to be used for training
    x_test : np.ndarray, n_test x 128
        Face encodings to be used for confidence evaluation
    y_train : np.ndarray, n_train x 1
        Labels corresponding to `x_train` to be used for training
    y_test : np.ndarray, n_test x 1
        Labels corresponding to `x_test` to be used for confidence evaluation

    """
    order = np.argsort(y)
    xo, yo = x[order], y[order]
    split_indices = np.cumsum(np.unique(yo, return_counts=True)[1])[:-1]
    iterator = zip(np.split(xo, split_indices), np.split(yo, split_indices))
    _x_train, _y_train, _x_test, _y_test = [], [], [], []

    for gx, gy in iterator:
        _order = dist_mat_order(gx)
        split_index = int(gx.shape[0] * (1 - train_fract))
        train_index, test_index = (
            _order[split_index:], _order[:split_index])

        _x_train.append(gx[train_index])
        _y_train.append(gy[train_index])
        _x_test.append(gx[test_index])
        _y_test.append(gy[test_index])

    __x_train, __y_train = np.concatenate(_x_train), np.concatenate(_y_train)
    train_perm = np.random.permutation(__x_train.shape[0])
    x_train, y_train = __x_train[train_perm], __y_train[train_perm]

    __x_test, __y_test = np.concatenate(_x_test), np.concatenate(_y_test)
    train_perm = np.random.permutation(__x_test.shape[0])
    x_test, y_test = __x_test[train_perm], __y_test[train_perm]

    return x_train, x_test, y_train, y_test
