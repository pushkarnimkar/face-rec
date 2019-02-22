from recon.jpeg.compressed import Compressed
from typing import Callable, Any

import numpy as np
import os


def is_jpeg(file_name: str):
    return file_name.endswith(".jpeg") or file_name.endswith(".jpg")


def map_dir(dir_name: str, func: Callable[[Compressed, str], Any]):
    jpeg_list = map(lambda name: os.path.join(dir_name, name),
                    filter(is_jpeg, os.listdir(dir_name)))
    for jpeg_name in jpeg_list:
        with open(jpeg_name, "rb") as jpeg_file:
            compressed = Compressed(jpeg_file)
            compressed.parse()
            result = func(compressed, jpeg_name)
        yield result


def extract_luminance(compressed: Compressed, jpeg_name: str):
    luminance: np.ndarray = np.array([mcu[0] for mcu in compressed.mcus[1]])
    return luminance.reshape(60, 80), jpeg_name, compressed.quant_tbl[1]
