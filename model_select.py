from keras.layers import Dense, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from functools import partial
from typing import Tuple

import argparse
import numpy as np
import os
import pandas as pd

from image_store import ImageStore
from ask import convex_hull_sequence, random_sequence


def make_model(subs_count: int, weights_tmp_file: str) -> Model:
    # weights_tmp_file = "/tmp/weights.hdf5"

    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=128))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(subs_count, activation="softmax"))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])

    if os.path.exists(weights_tmp_file):
        model.load_weights(weights_tmp_file)
    else:
        model.save_weights(weights_tmp_file)

    return model


def train_and_evaluate(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                       y_test: np.ndarray, weights_tmp_file: str,
                       subs_count: int) -> Tuple[np.float, np.float]:

    model = make_model(subs_count, weights_tmp_file)
    model.fit(x_train, to_categorical(y_train, subs_count),
              epochs=1000, batch_size=50, verbose=0)
    return model.evaluate(x_test, to_categorical(y_test, subs_count))


def eval_seq(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
             y_test: np.ndarray, seq: np.ndarray,
             weights_tmp_file: str, method: str) -> dict:

    subs_count = np.unique(np.concatenate((y_train, y_test))).shape[0]
    fractions, scores = \
        list(map(partial(round, ndigits=4), np.arange(0.0, 1.05, 0.05))), {}

    for fraction in fractions:
        split_point = int(fraction * seq.shape[0])
        x_train_fract, y_train_fract = (x_train[seq[:split_point], :],
                                        y_train[seq[:split_point]])
        score = train_and_evaluate(x_train_fract, y_train_fract, x_test, y_test,
                                   weights_tmp_file, subs_count)
        scores[fraction] = score
        log_string = f'method={method}, fraction={fraction}, ' \
                     f'loss={score[0]}, acc={score[1]}'
        print(log_string)
    return scores


def auc_alc(values: np.ndarray, intv: float):
    """Area under active learners' curve"""
    if values.shape[0] <= 2:
        raise ValueError("auc evaluation requires at-least two values")
    return intv * (0.5 * (values[0] + values[-1]) + values[1:-1].sum())


def train_test_split(x: np.ndarray, y: np.ndarray, train_fract: float) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    count = x.shape[0]
    train_count = int(train_fract * count)

    perm = np.random.permutation(count)
    x_train, y_train = x[perm[:train_count]], y[perm[:train_count]]
    x_test, y_test = x[perm[train_count:]], y[perm[train_count:]]
    return x_train, y_train, x_test, y_test


def find_scores(store_dir: str, methods: dict, weights_tmp_file: str= "/tmp/weights.hdf5",
                train_fract: float=0.6) -> dict:

    store, scores = ImageStore.read(store_dir), {}
    x, y = store.encs, pd.Categorical(store.info["subject"]).codes
    x_train, y_train, x_test, y_test = train_test_split(x, y, train_fract)

    for method, func in methods.items():
        seq = func(x_train)
        assert set(seq) == set(np.arange(x_train.shape[0]))
        scores[method] = eval_seq(x_train, y_train, x_test, y_test, seq,
                                  weights_tmp_file, method)

    os.remove(weights_tmp_file)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("store_dir", dict(random=random_sequence,
                                          convex_hull=convex_hull_sequence))

    args = parser.parse_args()
    final_scores = find_scores(args.store_dir)
    print(final_scores)
