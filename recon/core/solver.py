from algos.utils import simplifier
from typing import Optional, Tuple
from recon.app.image_store import ImageStore
from recon.core.estimator import (PredictorModel, ConfidenceModel,
                                  load_predictor_model, load_confidence_model)
from recon.data.pool import POOLTYPE, transform_pool

import numpy as np


class Solver:
    def __init__(self, pool: Optional[POOLTYPE]=None,
                 stored: Optional[dict]=None,
                 local: bool=True,
                 confidence_method: Optional[str]="zscore",
                 predictor_method: Optional[str]="neural"):

        self.predictor_method = predictor_method
        self.confidence_method = confidence_method

        self.predictor_model: Optional[PredictorModel] = None
        self.confidence_model: Optional[ConfidenceModel] = None
        self.mapping: Optional[np.ndarray] = None
        self.is_local: bool = local
        self.fitted = False

        if pool is None and stored is None:
            pass
        elif pool is not None:
            self._build_solver(pool)
        elif stored is not None:
            self._load_solver(stored)

    def _transform(self, pool: POOLTYPE) -> \
            Tuple[np.ndarray, Optional[np.ndarray]]:

        if isinstance(pool, tuple) and len(pool) == 2:
            return pool
        if not self.is_local:
            encodings, labels = transform_pool(pool)
            return encodings, labels
        elif self.is_local and isinstance(pool, ImageStore):
            return pool.get_verified()
        elif self.is_local and isinstance(pool, dict):
            return (pool["encodings"].reshape(-1, 1),
                    np.array([pool["subject"]]))
        elif self.is_local and isinstance(pool, np.ndarray):
            return pool.reshape(-1, 128), None

    def _build_solver(self, pool: POOLTYPE):
        encodings, subjects = self._transform(pool)
        self.mapping, labels = np.unique(subjects, return_inverse=True)

        self.predictor_model = load_predictor_model(
            self.predictor_method, subject_count=self.mapping.shape[0])
        self.confidence_model = load_confidence_model(self.confidence_method)

        x_train, x_test, y_train, y_test = \
            self.confidence_model.split(encodings, labels, policy="dmotli")
        try:
            self.predictor_model.fit(x_train, y_train)
            self.confidence_model.fit(self.predictor_model, x_test, y_test)
            self.fitted = True
        except ValueError:
            self.fitted = False

    def _load_solver(self, model_json: dict):
        _deserialized = simplifier.deserialize(model_json)
        self.predictor_model = _deserialized["predictor_model"]
        self.confidence_model = _deserialized["confidence_model"]
        self.mapping = np.array(_deserialized["mapping"])
        self.fitted = True

    def recognize(self, pool: POOLTYPE) -> Tuple[np.ndarray, np.ndarray]:
        encodings, _ = self._transform(pool)
        _prediction_prob = self.predictor_model.predict(encodings)
        confidence = self.confidence_model.evaluate(_prediction_prob)
        _prediction = np.argmax(_prediction_prob, axis=1)
        prediction = self.mapping[_prediction]
        return confidence, prediction

    def export_model(self):
        if not self.fitted:
            return None
        _export = dict(predictor_model=self.predictor_model,
                       confidence_model=self.confidence_model,
                       mapping=self.mapping.tolist())
        return simplifier.serialize(_export)
