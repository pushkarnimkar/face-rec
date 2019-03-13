from typing import Optional

import numpy as np


class BaseSequencer:
    name = "base_sequencer"

    def sequence(self, encs: np.ndarray) -> np.ndarray:
        return np.arange(encs.shape[0])


class RandomSequencer(BaseSequencer):
    name = "random"

    def __init__(self, seed: Optional[int]=None):
        self.seed = seed

    def sequence(self, encs: np.ndarray) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.permutation(encs.shape[0])
