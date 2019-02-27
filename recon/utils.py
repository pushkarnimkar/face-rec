from recon.jpeg.compressed import Compressed
from typing import Callable

import cv2
import numpy as np


def down_sample(file_name: str, ratio: int=4):
    compressed = Compressed(open(file_name, "rb"))
    compressed.parse()
    return compressed.down_sample(ratio=ratio)


def sobel_norm(src):
    sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
    return np.linalg.norm(np.dstack((sobelx, sobely)), axis=2)


def grid_apply(src: np.ndarray, cell_width: int, cell_height: int,
               stride_width: int, stride_height: int,
               func: Callable[[np.ndarray], np.ndarray]):

    sep_x, sep_y = cell_width - stride_width, cell_height - stride_height
    xs0 = np.arange(0, src.shape[0], sep_x)
    xs1 = np.arange(cell_width, src.shape[0], sep_x)
    xs0 = xs0[0:xs1.shape[0]]

    ys0 = np.arange(0, src.shape[1], sep_y)
    ys1 = np.arange(cell_height, src.shape[1], sep_y)
    ys0 = ys0[0:ys1.shape[0]]

    buffer = np.zeros((xs0.shape[0], ys0.shape[0]))
    for xi in range(xs0.shape[0]):
        for yi in range(ys0.shape[0]):
            val = func(src[xs0[xi]:xs1[xi], ys0[yi]:ys1[yi]])
            buffer[xi, yi] = val

    return buffer

