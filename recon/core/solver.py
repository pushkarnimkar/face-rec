from algos.utils import simplifier
from typing import Optional, Tuple
from recon.app.image_store import ImageStore
from recon.core.estimator import (PredictorModel, ConfidenceModel,
                                  load_predictor_model, load_confidence_model)
from recon.data.pool import POOLTYPE, transform_pool

import numpy as np


class FRSolver:
    def __init__(self, pool: Optional[POOLTYPE]=None,
                 stored: Optional[dict]=None,
                 local: bool=True,
                 confidence_method: Optional[str]="zscore",
                 predictor_method: Optional[str]="neural"):

        assert not (pool is None and stored is None), \
            "both of pool and model_json can not be NoneType"

        self.predictor_model: PredictorModel = \
            load_predictor_model(predictor_method)
        self.confidence_model: ConfidenceModel = \
            load_confidence_model(confidence_method)

        self.mapping: Optional[np.ndarray] = None
        self.is_local: bool = local

        if pool is not None:
            self._build_solver(pool)

        elif stored is not None:
            self._load_solver(stored)

    def _transform(self, pool: POOLTYPE) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_local:
            encodings, labels = transform_pool(pool)
            return encodings, labels
        elif self.is_local and isinstance(pool, ImageStore):
            return pool.encs, pool.info["subject"].values
        elif self.is_local and isinstance(pool, dict):
            return (pool["encodings"].reshape(-1, 1),
                    np.array([pool["subject"]]))

    def _build_solver(self, pool: POOLTYPE):
        encodings, subjects = self._transform(pool)
        self.mapping, labels = np.unique(subjects, return_inverse=True)
        x_train, x_test, y_train, y_test = \
            self.confidence_model.split(encodings, labels)
        self.predictor_model.fit(x_train, y_train)
        self.confidence_model.fit(self.predictor_model, x_test, y_test)

    def _load_solver(self, model_json: dict):
        _deserialized = simplifier.deserialize(model_json)
        self.predictor_model = _deserialized["predictor_model"]
        self.confidence_model = _deserialized["confidence_model"]
        self.mapping = np.array(_deserialized["mapping"])

    def recognize(self, pool: POOLTYPE) -> Tuple[np.ndarray, np.ndarray]:
        encodings, _ = self._transform(pool)
        _prediction_prob = self.predictor_model.predict(encodings)
        confidence = self.confidence_model.evaluate(_prediction_prob)
        _prediction = np.argmax(_prediction_prob, axis=0)
        prediction = self.mapping[_prediction]
        return confidence, prediction

    def export_model(self):
        _export = dict(predictor_model=self.predictor_model,
                       confidence_model=self.confidence_model,
                       mapping=self.mapping.tolist())
        return simplifier.serialize(_export)
