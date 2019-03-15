from joblib import Parallel, delayed
from keras.layers import Dense, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from functools import partial
from tqdm import tqdm
from typing import Tuple, List

import argparse
import numpy as np
import pandas as pd
import pickle

from recon.core.sequence import (
    BaseSequencer, ConvexHullSequencer, RandomSequencer,
    IterativeHullSequencer, DistanceMatrixSequencer
)
from helpers import train_test_split
from recon.app.image_store import ImageStore


def make_model(subs_count: int) -> Model:
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=128))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(subs_count, activation="softmax"))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    return model


def train_and_evaluate(x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray, y_test: np.ndarray,
                       subs_count: int) -> Tuple[np.float, np.float]:
    model = make_model(subs_count)
    _y_train, _y_test = (to_categorical(y_train, subs_count),
                         to_categorical(y_test, subs_count))
    model.fit(x_train, _y_train, epochs=1000, batch_size=100, verbose=0)
    return model.evaluate(x_test, _y_test, verbose=0)


def eval_seq(x_train: np.ndarray, y_train: np.ndarray,
             x_test: np.ndarray, y_test: np.ndarray,
             seq: np.ndarray, method: str, progress=True) -> pd.DataFrame:
    subs_count = np.unique(np.concatenate((y_train, y_test))).shape[0]
    fractions, logs = \
        list(map(partial(round, ndigits=4), np.arange(0, 1.01, 0.05))), []
    iterator = tqdm(fractions, desc=method) if progress else fractions

    for fraction in iterator:
        split_point = int(fraction * seq.shape[0])
        x_train_fract, y_train_fract = (x_train[seq[:split_point], :],
                                        y_train[seq[:split_point]])
        score = train_and_evaluate(x_train_fract, y_train_fract,
                                   x_test, y_test, subs_count)
        logs.append((method, fraction, score[0], score[1], split_point))

    frame = pd.DataFrame(logs, columns=["method", "fraction", "loss",
                                        "accuracy", "split_point"])
    return frame


def auc_alc(vals: np.ndarray, step: float) -> np.float64:
    """
    Area under curve of active learning curve performance metric of
    an active learner.

    Parameters
    ----------
    vals : numpy array of accuracy values with respect to fraction
    step : step size of fraction

    Returns
    -------
    score : area under curve score of active learner
    """
    if vals.shape[0] <= 2:
        raise ValueError("evaluation requires minimum two values")
    return step * ((vals[0] + vals[-1]) / 2 + vals[1:-1].sum())


def evaluate_sequencer(sequencer: BaseSequencer,
                       x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray, y_test: np.ndarray):
    sequence = sequencer.sequence(x_train)
    assert set(sequence) == set(np.arange(x_train.shape[0]))
    return eval_seq(x_train, y_train, x_test, y_test, sequence,
                    sequencer.name, progress=False)


def find_scores(store_dir: str, methods: List[BaseSequencer],
                train_fract: float=0.6, repeat: int=10) -> List[pd.DataFrame]:
    """
    Runs each of the `methods` for `repeat` number of iterations and by
    calling the splitter for each repetition on `ImageStore` located at
    `store_dir`. Returns results as list of data frames.

    Parameters
    ----------
    store_dir : path
        Path of `ImageStore` for reading encodings
    methods : List[BaseSequencer]
        List containing instances of BaseSequencer (potentially different
        sequencing algorithms with differing arguments)
    train_fract : float (default = 0.6)
        Fraction of encoding data to be used as training set
    repeat : int
        Number of times to repeat split process to generate different splits
        so as to evaluate performance of algorithm robustly.

    Returns
    -------
    scores : List[DataFrame]
        List containing different data frames

    """
    store, scores = ImageStore.read(store_dir), {}
    x, y = store.encs, pd.Categorical(store.info["subject"]).codes
    scores = []
    index_columns = ["method", "fraction"]

    with Parallel(n_jobs=len(methods)) as parallel:
        for _ in range(repeat):
            x_train, y_train, x_test, y_test = \
                train_test_split(x, y, train_fract)
            evaluate = partial(evaluate_sequencer, x_train=x_train,
                               y_train=y_train, x_test=x_test, y_test=y_test)
            _scores = parallel(
                delayed(evaluate)(sequencer) for sequencer in methods)
            scores.append(
                pd.concat(_scores).set_index(index_columns, drop=True))
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("store_dir")

    args = parser.parse_args()
    _methods = [
        DistanceMatrixSequencer(weight=0.3), ConvexHullSequencer(),
        DistanceMatrixSequencer(weight=1.0), RandomSequencer(),
        DistanceMatrixSequencer(weight=3.0), IterativeHullSequencer(),
    ]
    final_scores = find_scores(args.store_dir, _methods, repeat=10)
    with open("/tmp/scores.pickle", "wb") as out:
        pickle.dump(final_scores, out)

    frame = pd.concat(final_scores) \
        .groupby(level=["method", "fraction"]) \
        .aggregate(["count", "mean", "std"])

    print(frame[["accuracy", "loss"]].loc[(slice(None), [0.25, 0.5, 0.75]), ])
