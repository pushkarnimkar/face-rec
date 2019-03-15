from enum import Enum
from face_recognition import face_encodings, face_locations
from typing import Optional, Tuple, Union

import base64
import cv2
import hashlib
import numpy as np


class TransformFailure(Enum):
    INVALID_JPEG = "INVALID_JPEG"
    FACE_DETECTION_FAILED = "FACE_DETECTION_FAILED"


TRANSFORM_RETT = Union[Tuple[np.ndarray, np.ndarray, np.ndarray, str],
                       None, TransformFailure]


class TransformError(TypeError):
    def __init__(self, code: TransformFailure):
        self.code = code


def transform_image(image_input: Union[str, bytes]) -> np.ndarray:
    """
    Handles transformation of image in various formats like jpeg, png, etc
    and encoded in either binary or base64 format to array of R G B values.

    Parameters
    ----------
    image_input : Image in base64 string or bytes format

    Returns
    -------
    numpy array of R G B values of shape (height, width, 3)

    Raises
    ------
    TypeError : When input is not a valid jpeg buffer

    """
    buffer = (base64.b64decode(image_input)
              if isinstance(image_input, str) else image_input)
    image_compressed = np.frombuffer(buffer, np.int8)
    image_bgr = cv2.imdecode(image_compressed, cv2.IMREAD_COLOR)
    if image_bgr is not None:
        image_array = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_array
    else:
        raise TransformError(TransformFailure.INVALID_JPEG)


def extract_features(image: np.ndarray, face_location_model: str="hog") -> \
        Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Extracts bounding box and `face_recognition` encoding of largest face in
    the image and md5 hash of image as numpy array

    Parameters
    ----------
    image: numpy array R G B values
    face_location_model: Parameter to face_locations call. Refer to:
        https://face-recognition.readthedocs.io/en/latest/face_recognition.html#face_recognition.api.face_locations

    Returns
    -------
    image_hash: MD5 hash of numpy array of image
    face_box: coordinates of bounding box of face in image (x1, y1, x2, y2)
    face_encodings: vector encoding of face as per
        `face_recognition` library

    """
    bbox = np.array(face_locations(image, model=face_location_model))
    if bbox.shape[0] == 0:
        raise TransformError(TransformFailure.FACE_DETECTION_FAILED)
    area = np.abs((bbox[:, 0] - bbox[:, 2]) *
                  (bbox[:, 1] - bbox[:, 3]))
    image_hash = hashlib.md5(image).hexdigest()
    face_box = bbox[np.argmax(area), :]
    face_encoding = face_encodings(image, [face_box])[0]
    return image_hash, face_box, face_encoding


def transform(image_input: Union[str, bytes], face_location_model: str="hog",
              error_code: bool=False) -> TRANSFORM_RETT:
    """
    Transforms image buffer into face parameters and hash of of the image.
    If fails to find face, returns None.

    Parameters
    ----------
    image_input : (string | bytes)
        Image in base64 string or bytes format
    face_location_model : str, default "hog"
        Model for face detection used by `face_recognition` library
    error_code : bool, default false
        If this is set returns error code on failure instead of `None`

    Returns
    -------
    image : np.ndarray, success
        Image as an array of shape: (width, height, channels)
    face_encodings : np.ndarray, success
        Vector encoding of face as per `face_recognition` library
    face_box : np.ndarray, success
        Coordinates of bounding box of face in image (x1, y1, x2, y2)
    image_hash : str, success
        MD5 hash of numpy array of image
    error_code : (None | TransformFailure)
        Code describing reason of failure if `error_code` is set in input,
        else returns `None`.

    """
    try:
        image = transform_image(image_input)
        image_hash, face_box, face_encoding = extract_features(
            image, face_location_model=face_location_model)
        return image, face_encoding, face_box, image_hash
    except TransformError as te:
        return te.code if error_code else None
