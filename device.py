import cv2
import numpy as np


def server_acquire_image() -> np.ndarray:
    image = cv2.imread("static/img/demo.jpg")
    return image
