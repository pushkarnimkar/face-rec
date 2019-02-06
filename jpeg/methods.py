from itertools import product
from jpeg.compressed import Compressed
from jpeg.markers import QuantizationTable
from scipy.fftpack import idctn
from typing import Callable, Any, List

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


def decompress_mcu(mcu: np.ndarray, quant_tbl: QuantizationTable):
    coefs = mcu[quant_tbl.order] * quant_tbl.table
    return idctn(coefs)


def decompress_mcus(mcus: List[np.ndarray], offset_x: int, offset_y: int,
                    height: int, width: int, quant_tbl: QuantizationTable):
    buffer = np.zeros((8 * height, 8 * width))
    for mcux, mcuy in product(range(width), range(height)):
        mcu_index = (offset_y + mcuy) * 80 + (offset_x + mcux)
        buffer[mcuy * 8: (mcuy + 1) * 8, mcux * 8: (mcux + 1) * 8] = \
            decompress_mcu(mcus[mcu_index], quant_tbl)
    return buffer


def decompress(compressed: Compressed):
    quant_tbl = compressed.scan_headers[0].components[1].quant_tbl
    return decompress_mcus(compressed.mcus[1], 0, 0, 60, 80, quant_tbl)
