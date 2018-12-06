from image_store import ImageStore

from typing import Tuple, Union

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical

import cv2
import numpy as np
import pandas as pd
import face_recognition

import os
import sys
import time


class EncodingsClassifier:
    def __init__(self, model_dir: str="", encs: np.ndarray=None,
                 subs: np.ndarray=None, force_train: bool=False):

        if encs is None or subs is None:
            self.is_none = True
            return

        self.is_none = False
        subs_count = np.unique(subs).shape[0]
        self.__create_model__(subs_count)
        weights_file = os.path.join(model_dir, "weights.json")

        if os.path.exists(weights_file) and not force_train:
            self.__load_model__(model_dir)
        else:
            self.__train_model__(encs, subs)

        # history = model.fit(x_train, y_train, epochs=2000, batch_size=50)

    def __create_model__(self, subs_count):
        self.model = Sequential()
        self.model.add(Dense(64, activation="relu", input_dim=128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(subs_count, activation="softmax"))

    def __load_model__(self):
        pass

    def __train_model__(self, encs: np.ndarray, subs: np.ndarray):
        train_index = pd.Series(subs).groupby(subs)\
            .apply(lambda g: g.iloc[:4])\
            .reset_index(level=0, drop=True).index.values

        train_mask, test_mask = (np.repeat(False, subs.shape[0]),
                                 np.repeat(True, subs.shape[0]))
        train_mask[train_index], test_mask[train_index] = True, False
        subs_count = np.unique(subs).shape[0]

        x_train = encs[train_mask, :]
        y_train = to_categorical(subs[train_mask], num_classes=subs_count)
        x_test = encs[test_mask, :]
        y_test = to_categorical(subs[test_mask], num_classes=subs_count)

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=1000, batch_size=50)
        score = self.model.evaluate(x_test, y_test)

        pred = self.model.predict(encs)
        self.__make_conf_model__(pred, subs)
        print(score)

    def __make_conf_model__(self, pred, subs):
        pred_subs = np.argmax(pred, axis=1)
        mask = pred_subs == subs
        correct, softmax_stats = pred[mask, :], {}

        for sub in subs:
            sub_mask = subs == sub
            sub_pred = correct[sub_mask[mask], sub]
            softmax_stats[sub] = (sub_pred.mean(), sub_pred.std())

        self.softmax_stats = softmax_stats

    def __conf_eval__(self, pred: np.ndarray) -> float:
        sub = np.argmax(pred)
        prob, (mean, std) = pred[0][sub], self.softmax_stats[sub]
        confidence = np.exp((prob - mean) / std)
        return confidence if confidence < 1 else 1

    def predict(self, enc: np.ndarray) -> (np.ndarray, np.ndarray):
        if self.is_none:
            return None, 0, 0

        pred = self.model.predict(enc.reshape(1, -1))
        pred_sub = np.argmax(pred, axis=1)[0]
        return pred_sub, self.__conf_eval__(pred), pred[0, pred_sub]


class FaceRecognizer:
    def __init__(self, store_dir: str=None, image_store: ImageStore=None,
                 confidence_thresh=0.75):

        if image_store is not None:
            self.store = image_store
        elif store_dir is not None:
            if len(os.listdir(store_dir)) > 1:
                self.store = ImageStore.read(store_dir)
            else:
                self.store = ImageStore(store_dir)
        else:
            raise ValueError("could not initialize image store")

        self.classifier = EncodingsClassifier()
        self.confidence_thresh = confidence_thresh

    @classmethod
    def encode(cls, img: np.ndarray) -> Tuple[np.ndarray, tuple]:
        """Finds and encodes largest face in the image"""
        boxes = np.array(face_recognition.face_locations(img))
        if len(boxes) == 0:
            return None, None

        delta_x, delta_y = boxes[:, 0] - boxes[:, 2], boxes[:, 1] - boxes[:, 3]

        face_idx = np.argmax(np.abs(delta_x * delta_y))
        face_box = boxes[face_idx, :]

        encoded = face_recognition.face_encodings(img, [face_box])[0]
        return encoded, face_box

    def feed(self, image: np.ndarray, vid: str, cap_time: int,
             force: bool=False) -> Tuple[Union[str, None], Union[dict, None]]:

        enc, box = self.encode(image)

        if enc is None or box is None:
            return None, None

        pred, conf, prob = self.classifier.predict(enc)

        if conf < self.confidence_thresh or force:
            sub = None if pred is None else self.subs_lst[pred]
            name = self.store.add(image, enc, box, vid, cap_time, sub, conf)
            return name, dict(box=box.tolist())
        else:
            return None, dict(box=box.tolist(), sub=self.subs_lst[pred],
                              conf=conf)

    def train(self):
        mask = self.store.info["verified"].values
        encs = self.store.encs[mask]
        subs = self.store.info["subject"][mask].map(self.subs_map).values
        self.classifier = EncodingsClassifier(encs=encs, subs=subs)

    def tell(self, name: str, subject: str):
        cols = ["subject", "confidence", "verified"]
        if name in self.store.info.index:
            self.store.info.loc[name, cols] = (subject, 1, True)

    def ask(self) -> Tuple[Union[str, None], Tuple[np.ndarray, dict]]:
        unverified = self.store.info[~self.store.info["verified"]]
        if unverified.shape[0] != 0:
            name = unverified.sample(1).index[0]
            return name, self.store.get_image(name, True)
        else:
            return None, (np.array([]), dict())

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
    def subs_map(self):
        return {sub: i for i, sub in enumerate(self.subs_lst)}


def main():
    if os.path.exists(os.path.join(sys.argv[2], ImageStore.INFO_FILE_NAME)):
        recognizer = FaceRecognizer(sys.argv[2])
    else:
        recognizer = FaceRecognizer.build(sys.argv[1], sys.argv[2], False)

    img = cv2.cvtColor(cv2.imread(sys.argv[3]), cv2.COLOR_BGR2RGB)
    img_name, pred = recognizer.feed(img, "testing", int(time.time()))

    query_name, (img, info) = recognizer.ask()
    print("predicted subject: " + pred)

    recognizer.store.write()


if __name__ == "__main__":
    main()
