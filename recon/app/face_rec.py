from recon.core.sequence import ConvexHullSequencer
from recon.app.image_store import ImageStore
from recon.core.solver import Solver
from recon.core.transform import transform
from typing import Tuple, Union, Iterator, Optional

import cv2
import json
import numpy as np
import os
import tensorflow as tf
import time


class FaceRecognizer:
    def __init__(self, store_dir: str, min_confidence: Optional[float]=0.75):
        if not os.path.exists(store_dir):
            raise FileNotFoundError(f"{store_dir} does not exist")

        self.stored_model_path = os.path.join(store_dir, "model.json")
        self.store = ImageStore.read(store_dir)
        self.solver: Optional[Solver] = None
        self.confidence_thresh = min_confidence
        self.iter_ask = self._ask()
        self.graph: Optional[tf.Graph] = None

    def feed(self, buffer: bytes, vid: str, cap_time: int,
             force: bool=False) -> Tuple[Optional[str], Optional[dict]]:
        if self.solver is None:
            raise ValueError(f"initialize the {self.__class__.__name__} first")

        try:
            image, enc, box, image_hash = \
                transform(buffer, face_location_model="cnn")
        except TypeError:
            return None, None

        with self.graph.as_default():
            _conf, _pred = self.solver.recognize(enc)

        conf, pred = _conf[0], _pred[0]
        value = dict(box=box.tolist(), sub=pred, conf=conf, image=image)
        if conf < self.confidence_thresh or force:
            name = self.store.add(image, enc, box, vid, cap_time, pred,
                                  conf, image_hash=image_hash)
            return name, value
        else:
            return None, value

    def train(self):
        if self.graph is not None:
            with self.graph.as_default():
                self.solver = Solver(pool=self.store)
        else:
            self.solver = Solver(pool=self.store)
            self.graph = tf.get_default_graph()
        stored_model = self.solver.export_model()
        with open(self.stored_model_path, "w") as model_file:
            json.dump(stored_model, model_file)

    def load(self):
        with open(self.stored_model_path) as model_file:
            _stored = json.load(model_file)
        self.solver = Solver(stored=_stored)
        self.graph = tf.get_default_graph()

    def tell(self, name: str, subject: str):
        cols = ["subject", "confidence", "verified"]
        if name in self.store.info.index:
            self.store.info.loc[name, cols] = (subject, 1, True)

    def _ask(self, method="input_order") -> \
            Iterator[Tuple[str, Tuple[np.ndarray, dict]]]:

        index = np.where(~self.store.info["verified"])[0]

        if method == "input_order":
            sequence = index
        else:
            unverified_encs = self.store.encs[index, :]
            if callable(method):
                sequence = index[method(unverified_encs)]
            elif method == "convex_hull":
                sequencer = ConvexHullSequencer()
                sequence = index[sequencer.sequence(unverified_encs)]
            else:
                raise ValueError("method not understood")

        if sequence.shape[0] == 0:
            return
        for seq_num in sequence:
            info = self.store.info.iloc[seq_num]
            yield info.name, self.store.get_image(info.name, True)

    def ask(self) -> Union[Tuple[str, Tuple[np.ndarray, dict]], None]:
        asked = None
        try:
            asked = next(self.iter_ask)
        except (TypeError, StopIteration):
            self.iter_ask = self._ask()
            asked = next(self.iter_ask)
        finally:
            return asked

    def feed_again(self):
        for buffer in self.store.absent_images():
            self.feed(buffer, "feed_again", int(time.time()), force=True)

    @staticmethod
    def build(train_dir: str, store_dir: str) -> "FaceRecognizer":

        assert len(os.listdir(store_dir)) == 1, \
            "build method expects empty store"
        recognizer = FaceRecognizer(store_dir)
        jpegs = filter(lambda n: n.endswith(".jpg"), os.listdir(train_dir))
        names = map(lambda n: (n, n[:-4].split("_")[0]), jpegs)

        for name, subject in names:
            img = cv2.cvtColor(cv2.imread(os.path.join(train_dir, name)),
                               cv2.COLOR_BGR2RGB)
            vid, timestamp = "training", int(time.time())
            name, _ = recognizer.feed(img, vid, timestamp)
            recognizer.tell(name, subject)

        recognizer.train()
        return recognizer

    @property
    def subs_lst(self):
        return list(self.store.info["subject"].unique())

    @property
    def has_stored_model(self):
        return os.path.exists(self.stored_model_path)
