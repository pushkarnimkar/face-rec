from abc import ABC, abstractmethod
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from typing import Callable, Optional

import numpy as np


class PredictorModel(ABC):
    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass


class ConfidenceModel(ABC):
    @classmethod
    def split(cls, x: np.ndarray, y: np.ndarray, train_fract: float = 0.6):
        take_train = int(x.shape[0] * train_fract)
        prerm = np.random.permutation(x.shape[0])
        x_train, x_test = x[prerm[:take_train], :], x[prerm[take_train:], :]
        y_train, y_test = y[prerm[:take_train]], y[prerm[take_train:]]
        return x_train, x_test, y_train, y_test

    @abstractmethod
    def fit(self, predictor: PredictorModel, x: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        pass


def _build_fn(subs_cnt: int) -> Sequential:
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=128))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(subs_cnt, activation="softmax"))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    return model


class NeuralPredictorModel(PredictorModel):
    def __init__(self, subs_cnt: Optional[int]=0,
                 build_fn: Optional[Callable[[int], Sequential]]=None):
        if subs_cnt <= 0 or not isinstance(subs_cnt, int):
            raise ValueError(f"expected positive integer found {subs_cnt}")
        if build_fn is None:
            build_fn = _build_fn
        self.model = build_fn(subs_cnt)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(x_train, y_train, epochs=1000, batch_size=50)

    def predict(self, x: np.ndarray):
        return self.model.predict_proba(x)


class ZscoreConfidenceModel(ConfidenceModel):
    def __init__(self, min_count: Optional[int]=4):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.count: Optional[np.ndarray] = None
        self.min_count = min_count

    def fit(self, predictor: PredictorModel, x: np.ndarray, y: np.ndarray):
        _predictions = predictor.predict(x)
        _subjects = np.argmax(_predictions, axis=1)
        mask = np.equal(_subjects, y)
        predictions, labels, subjects = \
            _predictions[mask, :], y[mask], _subjects[mask]

        _count = predictions.shape[1]
        self.mean, self.std = np.zeros((_count,)), np.zeros((_count,))

        for subject in range(_count):
            _mask = np.equal(subjects, subject)
            _subject_predictions = predictions[_mask, subject]
            self.mean[subject] = _subject_predictions.mean()
            self.std[subject] = _subject_predictions.std()
            self.count[subject] = _subject_predictions.shape[0]

    def evaluate(self, predictions: np.ndarray):
        _subjects = np.argmax(predictions, axis=1)
        valid, prob = \
            self.count >= self.min_count, predictions[:, _subjects]
        _confidence = np.clip(np.exp((prob - self.mean) / self.std), 0, 1)
        _confidence[~valid] = 0
        return _confidence


def load_confidence_model(method: Optional[str]=None,
                          **kwargs) -> ConfidenceModel:
    if method == "zscore":
        return ZscoreConfidenceModel(**kwargs)
    raise ValueError(f"invalid method {method}")


def load_predictor_model(method: Optional[str]=None,
                         **kwargs) -> PredictorModel:
    if method == "neural":
        assert "subs_cnt" in kwargs, "subject count is required"
        return NeuralPredictorModel(**kwargs)
    raise ValueError(f"invalid method {method}")
