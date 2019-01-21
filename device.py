from serial import Serial
from typing import Optional

import cv2
import numpy as np
import os
import stat


def check_usb_term(path: str) -> bool:
    try:
        char_dev = stat.S_ISCHR(os.stat(path).st_mode)
        return char_dev and "ttyUSB" in path
    except FileNotFoundError:
        return False


def find_camera_device() -> str:
    """
    Works on the assumption that you haven't attached any other device after
    attaching the camera

    Returns
    -------
    camera_device_path : str,
        Path of camera device in /dev directory under the assumption that
        camera is usb terminal device

    Raises
    ------
    IndexError
        When it fails to detect attached camera device on USB
    """
    try:
        usb_terms = list(filter(check_usb_term, map(
            lambda path: os.path.join("/dev", path), os.listdir("/dev"))))
        camera_device_path = usb_terms[-1]
        return camera_device_path
    except IndexError as ie:
        ie.args = ("could not detect usb camera",)
        raise ie


class USBCamera:
    def __init__(self, camera_path: Optional[str]=None, baudrate: int=115200,
                 image_buffer_size: int=15000):
        """
        Constructs USB camera object device with configurations given

        Parameters
        ----------
        camera_path : str, optional
            File system path of camera device

        baudrate : int, optional
            Baudrate to be used for serial communication,
            defaults to 115200

        image_buffer_size : int, optional
            Image buffer size as set by the device

        """
        if camera_path is None:
            camera_path = find_camera_device()
        self.camera_path = camera_path
        self.baudrate = baudrate
        self.image_buffer_size = image_buffer_size
        self._open()

    def _open(self):
        self.serial = Serial(baudrate=self.baudrate, port=self.camera_path)

    def _close(self):
        self.serial.close()

    def close(self):
        """Close the serial connection"""
        self._close()

    def acquire_image(self) -> np.ndarray:
        """
        Acquires image from USB camera.

        Returns
        -------
        image : np.ndarray
            Image decoded using cv2.imdecode.
            Thus, follows BGR format instead of RGB
        """
        self.serial.write(b"SEND")
        read_array = [self.serial.read()
                      for _ in range(self.image_buffer_size)]
        image_jpeg = np.frombuffer(b"".join(read_array), np.int8)
        image = cv2.imdecode(image_jpeg, cv2.IMREAD_COLOR)
        return image

    def __del__(self):
        self._close()


def main():
    camera = USBCamera()
    image = camera.acquire_image()
    cv2.imwrite("/tmp/image.jpg", image)


if __name__ == "__main__":
    main()
