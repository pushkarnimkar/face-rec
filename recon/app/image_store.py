from typing import Tuple, Optional

import cv2
import hashlib
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
            info: pd.DataFrame = info.astype(COLS_TYPE)
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
        if len(os.listdir(store_dir)) > 1:
            info = pd.read_csv(os.path.join(store_dir, cls.INFO_FILE_NAME))
            info = info.set_index("name")

            encs = np.load(os.path.join(store_dir, cls.ENCS_FILE_NAME))
            bbox = np.load(os.path.join(store_dir, cls.BBOX_FILE_NAME)) \
                .astype(np.int64)
            return ImageStore(store_dir, info, encs, bbox)
        else:
            return ImageStore(store_dir)

    def write(self):
        print("writing info file")
        self.info.to_csv(os.path.join(self.store_dir, self.INFO_FILE_NAME))
        print("saving face embeddings")
        np.save(os.path.join(self.store_dir, self.ENCS_FILE_NAME),
                self.encs, allow_pickle=False)
        print("saving face boxes")
        np.save(os.path.join(self.store_dir, self.BBOX_FILE_NAME),
                self.bbox, allow_pickle=False)

    def add(self, image: np.ndarray, enc: np.ndarray, box: np.ndarray,
            vid: str, cap_time: int, subject: str, confidence: float,
            verified: bool=False, image_hash: Optional[str]=None):

        name = image_hash if image_hash is not None \
            else hashlib.md5(image).hexdigest()

        if name not in self.info.index:
            save_path = os.path.join(self.store_dir, "images", name + ".jpg")
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

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

    def __add__(self, other: "ImageStore") -> "ImageStore":
        _self, _other = self.info.copy(), other.info.copy()
        _self["src"], _other["src"] = 0, 1
        _self["idx"], _other["idx"] = (
            np.arange(self.encs.shape[0]), np.arange(other.encs.shape[0]))

        _info = pd.concat([_self, _other]).drop_duplicates()
        _encs = np.ndarray((_info.shape[0], 128))
        _bbox = np.ndarray((_info.shape[0], 4))

        self_mask, other_mask = _info["src"] == 0, _info["src"] == 1
        _encs[self_mask] = self.encs[_info[self_mask]["idx"], :]
        _encs[other_mask] = other.encs[_info[other_mask]["idx"], :]
        _bbox[self_mask] = self.bbox[_info[self_mask]["idx"], :]
        _bbox[other_mask] = other.bbox[_info[other_mask]["idx"], :]
        _info = _info.drop(["src", "idx"], axis=1)
        return ImageStore(self.store_dir, _info, _encs, _bbox)

    def _absent_images(self):
        for image in os.listdir(self.img_dir):
            path = os.path.join(self.img_dir, image)
            if not os.path.isfile(path):
                continue
            if not (image.endswith(".jpeg") or image.endswith(".jpg")):
                continue
            index = image.split(".")[0]
            if index in self.info.index:
                continue
            yield path

    def absent_images(self):
        for path in self._absent_images():
            with open(path, "rb") as file:
                buffer = file.read()
            os.remove(path)
            yield buffer

    def clean_up(self):
        for path in self._absent_images():
            os.remove(path)

    def get_verified(self):
        mask = self.info["verified"].values
        return self.encs[mask], self.info["subject"][mask].values
