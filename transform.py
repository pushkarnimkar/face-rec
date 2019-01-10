from face_recognition import face_encodings, face_locations
from typing import Optional, Tuple, Union

import base64
import cv2
import hashlib
import numpy as np


def transform_image(image_input: Union[str, bytes]) -> np.ndarray:
    """Handles transformation of image in various formats like
    jpeg, png, etc and encoded in either binary or base64
    format to numpy array of R G B values.

    :param image_input: Image in base64 string or bytes format
    :return: image array in R G B format
    """
    buffer = (base64.b64decode(image_input)
              if isinstance(image_input, str) else image_input)
    image_compressed = np.frombuffer(buffer, np.int8)
    image_bgr = cv2.imdecode(image_compressed, cv2.IMREAD_COLOR)
    image_array = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_array


def extract_features(image: np.ndarray) -> \
        Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """Extracts bounding box and `face_recognition` encoding of
    largest face in the image and md5 hash of image as numpy array
    :param image: numpy array R G B values
    :return: (md5 hash of image, face bounding box, face encoding)
    if face detected, else None
    """
    bbox = np.array(face_locations(image, model="hog"))
    area = np.abs((bbox[:, 0] - bbox[:, 2]) *
                  (bbox[:, 1] - bbox[:, 3]))
    if area.shape[0] == 0:
        return
    image_md5 = hashlib.md5(image).hexdigest()
    face_box = bbox[np.argmax(area), :]
    face_encoding = face_encodings(image, [face_box])[0]
    return image_md5, face_box, face_encoding
