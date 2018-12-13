from typing import Tuple
from hashlib import md5

import cv2
import numpy as np
import os
import pandas as pd


COLS_TYPE = {"vid": str, "cap_time": int, "subject": str,
             "confidence": float, "verified": bool, "path": str}


class ImageStore:
    INFO_FILE_NAME = "info.csv"
    ENCS_FILE_NAME = "encs.npy"
    BBOX_FILE_NAME = "bbox.npy"

    def __init__(self, store_dir: str = os.getcwd(),
                 info: pd.DataFrame = pd.DataFrame(),
                 encs: np.ndarray = np.ndarray((0, 128), dtype=np.float64),
                 bbox: np.ndarray = np.ndarray((0, 4), dtype=np.int64)) -> None:

        assert info.shape[0] == encs.shape[0], \
            f"mismatching shapes of encodings and frame " \
            f"{encs.shape}, {info.shape}"

        assert encs.shape[1] == 128, f"expected 128d face encoding"

        if info.empty:
            info = pd.DataFrame([], columns=list(COLS_TYPE.keys()))
            info = info.astype(COLS_TYPE)
            info.index.name = "name"

        self.info = info
        self.encs = encs
        self.bbox = bbox

        if not os.path.exists(store_dir):
            os.mkdir(store_dir)
        self.store_dir = store_dir

        self.img_dir = os.path.join(store_dir, "images")
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

    def get_image(self, name: str, tag_face=False, width=2, color=(255, 0, 0),
                  get_rgb=False) -> Tuple[np.ndarray, dict]:

        info, idx = self.info.loc[name], self.info.index.get_loc(name)
        read_img = cv2.imread(info["path"])
        img = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB) if get_rgb else read_img

        if tag_face:
            pt0 = (self.bbox[idx, 1], self.bbox[idx, 0])
            pt1 = (self.bbox[idx, 3], self.bbox[idx, 2])
            cv2.rectangle(img, pt0, pt1, color, width)

        return img, info.to_dict()

    @classmethod
    def read(cls, store_dir: str) -> "ImageStore":
        info = pd.read_csv(os.path.join(store_dir, cls.INFO_FILE_NAME))
        info = info.set_index("name")

        encs = np.load(os.path.join(store_dir, cls.ENCS_FILE_NAME))
        bbox = np.load(os.path.join(store_dir, cls.BBOX_FILE_NAME))
        return ImageStore(store_dir, info, encs, bbox)

    def write(self):
        print("writing info file")
        self.info.to_csv(os.path.join(self.store_dir, self.INFO_FILE_NAME))
        print("saving face embeddings")
        np.save(os.path.join(self.store_dir, self.ENCS_FILE_NAME),
                self.encs, allow_pickle=False)
        print("saving face boxes")
        np.save(os.path.join(self.store_dir, self.BBOX_FILE_NAME),
                self.bbox, allow_pickle=False)

    def add(self, img: np.ndarray, enc: np.ndarray, box: tuple,
            vid: str, cap_time: int, subject: str,
            confidence: float, verified: bool=False):

        name = md5(img).hexdigest()

        if name not in self.info.index:
            save_path = os.path.join(self.store_dir, "images", name + ".jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            self.encs = np.vstack((self.encs, enc))
            self.bbox = np.vstack((self.bbox, box))

            row = (vid, cap_time, subject, confidence, verified, save_path)
            self.info.loc[name] = row

        return name

    def is_consistent(self):
        assert self.info.shape[0] == self.encs.shape[0], \
            f"mismatching shapes of encodings and frame " \
            f"{self.encs.shape}, {self.info.shape}"

        assert self.encs.shape[1] == 128, f"expected 128d face encoding"

    def __getitem__(self, item: np.ndarray):
        info = self.info.iloc[item]
        encs, bbox = self.encs[item, :], self.bbox[item, :]
        return ImageStore(self.store_dir, info, encs, bbox)
