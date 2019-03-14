from typing import Optional

import numpy as np


class BaseSequencer:
    _name_ = "BaseSequencer"

    def sequence(self, encs: np.ndarray) -> np.ndarray:
        return np.arange(encs.shape[0])

    @property
    def name(self):
        return self._name_


class RandomSequencer(BaseSequencer):
    _name_ = "RandomSequencer"

    def __init__(self, seed: Optional[int]=None):
        self.seed = seed

    def sequence(self, encs: np.ndarray) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.permutation(encs.shape[0])

    @property
    def name(self):
        _seed = "seed=" + "none" if self.seed is None else str(self.seed)
        return super(RandomSequencer, self).name + f"({_seed})"
