from serial import Serial, SerialException
from typing import Optional

import cv2
import numpy as np
import os
import stat
import sys


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
                 image_buffer_size: int=15000, timeout: float=10):
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

        timeout : float, optional
            Set a read timeout value. Defaults to 10 seconds

        """
        if camera_path is None:
            try:
                camera_path = find_camera_device()
            except IndexError:
                camera_path = None
                self.isavailable = False

        self.camera_path = camera_path
        self.baudrate = baudrate
        self.timeout = timeout
        self.image_buffer_size = image_buffer_size
        self.isopen = False

        try:
            if not hasattr(self, "isavailable"):
                self._open()
                self.isavailable = True
        except SerialException as se:
            print("Failed to open camera. Acquire method will not work")
            print(se)
            self.isavailable = False
        finally:
            self._close()
            print("created " + repr(self), file=sys.stderr)

    def _open(self):
        self.isopen = True
        self.serial = Serial(baudrate=self.baudrate, port=self.camera_path,
                             timeout=self.timeout)

    def _close(self):
        if self.isavailable and self.isopen:
            self.serial.close()
        self.isopen = False

    def close(self):
        """Close the serial connection"""
        self._close()

    def _acquire_buffer(self) -> bytes:
        if not self.isavailable:
            raise SerialException()

        self._open()
        self.serial.write(b"SEND")
        buffer = self.serial.read_until(b'\xff\xd9')
        self._close()
        return buffer

    def acquire_image(self) -> Optional[np.ndarray]:
        """
        Acquires image from USB camera.

        Returns
        -------
        image : np.ndarray
            Image decoded using cv2.imdecode.
            Thus, follows BGR format instead of RGB
        """
        try:
            _buffer = self._acquire_buffer()
            print("buffer length -> " + str(len(_buffer)))
            index_jpeg_start = _buffer.index(b'\xff\xd8\xff\xe0')
            buffer = _buffer[index_jpeg_start:]

            image_jpeg = np.frombuffer(buffer, np.int8)
            image = cv2.imdecode(image_jpeg, cv2.IMREAD_COLOR)
            return image
        except ValueError as ve:
            print(ve, file=sys.stderr)
            return

    def __del__(self):
        print("deleting " + repr(self), file=sys.stderr)
        self._close()


def main():
    camera = USBCamera()
    image = camera.acquire_image()
    cv2.imwrite("/tmp/image.jpg", image)


if __name__ == "__main__":
    main()
