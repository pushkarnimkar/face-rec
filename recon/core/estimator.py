from abc import ABC, abstractmethod
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
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
    def __init__(self, subject_count: Optional[int]=0,
                 build_fn: Optional[Callable[[int], Sequential]]=None):
        self.subject_count = subject_count
        self.build_fn = build_fn
        self.model: Optional[Sequential] = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        if self.subject_count <= 0 or not isinstance(self.subject_count, int):
            raise ValueError(
                f"expected positive integer found {self.subject_count}")
        if self.build_fn is None:
            self.build_fn = _build_fn
        self.model = self.build_fn(self.subject_count)
        _y_train = to_categorical(y_train, num_classes=self.subject_count)
        self.model.fit(x_train, _y_train, epochs=1000, batch_size=50)

    def predict(self, x: np.ndarray):
        if self.model is None:
            raise ValueError("model not fitted")
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
        self.count = np.zeros((_count,)).astype(np.int)

        for subject in range(_count):
            _mask = np.equal(subjects, subject)
            _subject_predictions = predictions[_mask, subject]
            if _subject_predictions.shape[0] == 0:
                continue
            self.mean[subject] = _subject_predictions.mean()
            self.std[subject] = _subject_predictions.std()
            self.count[subject] = _subject_predictions.shape[0]

    def evaluate(self, predictions: np.ndarray):
        _subjects, _valid = np.argmax(predictions, axis=1), \
                            self.count >= self.min_count
        prob, valid = predictions[:, _subjects].flatten(), _valid[_subjects]
        _confidence = (prob - self.mean[_subjects]) / self.std[_subjects]
        _confidence = np.clip(np.exp(_confidence), 0, 1)
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
        assert "subject_count" in kwargs, "subject count is required"
        return NeuralPredictorModel(**kwargs)
    raise ValueError(f"invalid method {method}")
