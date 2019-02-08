from collections import defaultdict
from itertools import product
from jpeg.common import BitReader, ByteReader
from jpeg.markers import (read_soi, read_eoi, QuantizationTable, HuffmanTable,
                          read_marker, FrameHeader, ScanHeader, ScanComponent)
from jpeg.scan import decode_dc, decode_ac
from scipy.fftpack import idctn
from typing import Dict, List, Optional, Union, Tuple

import numpy as np


def _read_comp(reader, state_dc: list, component: ScanComponent):
    dc_diff, comp_id = decode_dc(reader, component), component.component_id
    dc_coef = state_dc[comp_id] + dc_diff
    ac_coef = decode_ac(reader, component)
    state_dc[comp_id] = dc_coef
    return np.insert(ac_coef, 0, dc_coef)


def _scan_imcu(reader: BitReader, scan_header: ScanHeader, state_dc: list):
    components = scan_header.components
    try:
        mcu11 = _read_comp(reader, state_dc, components[1])
        mcu12 = _read_comp(reader, state_dc, components[1])
        mcu20 = _read_comp(reader, state_dc, components[2])
        mcu30 = _read_comp(reader, state_dc, components[3])
        return mcu11, mcu12, mcu20, mcu30
    except TypeError:
        return None


def _decompress_comp(mcu: np.ndarray, quant_tbl: QuantizationTable):
    coefs = mcu[quant_tbl.order] * quant_tbl.table
    return idctn(coefs, norm="ortho")


def _decompress_mcu(mcu: Union[np.ndarray,
                               Tuple[np.ndarray, np.ndarray, np.ndarray]],
                    components: Dict[int, ScanComponent],
                    comp_id: Optional[int]=None):
    if isinstance(mcu, np.ndarray):
        assert comp_id is not None
        quant_tbl = components[comp_id].quant_tbl
        return _decompress_comp(mcu, quant_tbl)
    else:
        return tuple(_decompress_comp(mcu, components[comp_id].quant_tbl)
                     for mcu, comp_id in zip(mcu, (1, 2, 3)))


class Compressed:
    def __init__(self, stream):
        self._buffer = stream.read()
        self.stream = ByteReader(self._buffer)
        self.quant_tbl: Dict[int, QuantizationTable] = {}
        self.dc_huff_tbl: Dict[int, HuffmanTable] = {}
        self.ac_huff_tbl: Dict[int, HuffmanTable] = {}
        self.app0 = None
        self.sof0 = None
        self.scan_headers: List[ScanHeader] = []
        self.mcus = {}
        self.imcus = []
        self.parse_success = False

    def parse(self):
        if len(self.imcus) != 0:
            message = "calling parse on already populated compressed object"
            raise RuntimeError(message)
        read_soi(self.stream)
        marker, content = read_marker(self.stream)
        while marker != "SOF0":
            if marker == "DQT":
                self.update_quant_tbl(content)
            elif marker == "APP0":
                self.app0 = content
            elif marker == "DHT":
                self.update_huff_tbl(content)
            marker, content = read_marker(self.stream)
        else:
            self.sof0 = FrameHeader(content)

        marker, content = read_marker(self.stream)
        if marker == "SOS":
            scan_header = ScanHeader(content, self.sof0.components,
                                     self.ac_huff_tbl, self.dc_huff_tbl,
                                     self.quant_tbl)
            self.scan_headers.append(scan_header)
            reader = BitReader(self.stream, bs=4)
            state_dc, mcus = [0, 0, 0, 0], defaultdict(list)
            for _ in range(2400):
                try:
                    self.imcus.append(reader.get_pos() + tuple(state_dc))
                    blocks = _scan_imcu(reader, scan_header, state_dc)
                    if blocks is None:
                        break
                    m11, m12, m20, m30 = blocks
                    mcus[1].extend((m11, m12))
                    mcus[2].append(m20)
                    mcus[3].append(m30)
                except Exception as exc:
                    raise exc
                finally:
                    self.mcus = dict(mcus)
            else:
                read_eoi(self.stream)
                self.parse_success = True

            if not self.parse_success:
                raise ValueError("invalid JPEG input")
        else:
            raise ValueError(f"expected start of segment found {marker}")

    def update_quant_tbl(self, content: bytes):
        table = QuantizationTable(content)
        self.quant_tbl[table.table_id] = table

    def update_huff_tbl(self, content: bytes):
        table = HuffmanTable(content)
        if table.is_ac:
            self.ac_huff_tbl[table.table_id] = table
        else:
            self.dc_huff_tbl[table.table_id] = table

    def read_mcu(self, x: int, y: int, comp: Optional[int]=None):
        if not self.parse_success:
            raise ValueError("reading from non-parsed object")

        i = (y * 80 + x)
        _i = i // 2
        pos, state = self.imcus[_i][:2], list(self.imcus[_i][2:])
        reader = BitReader(self.stream, bs=4)
        reader.set_ptr(pos)
        imcu = _scan_imcu(reader, self.scan_headers[0], state)
        if imcu is None:
            raise ValueError("empty imcu")

        m11, m12, m20, m30 = imcu
        components = self.scan_headers[0].components
        if comp is None and i % 2:
            return _decompress_mcu((m12, m20, m30), components)
        elif comp is None:
            return _decompress_mcu((m11, m20, m30), components)
        return _decompress_mcu(imcu[comp], components, comp_id=comp) \
            if comp >= 2 else \
            _decompress_mcu(imcu[i % 2], components, comp_id=1)

    def decompress_region(self, offset_x: int, offset_y: int, height: int, width: int, comp: int=1):
        buffer = np.zeros((8 * height, 8 * width))
        for mcux, mcuy in product(range(width), range(height)):
            mcu = self.read_mcu(offset_x + mcux, offset_y + mcuy, comp)
            buffer[mcuy * 8: (mcuy + 1) * 8, mcux * 8: (mcux + 1) * 8] = mcu
        return buffer
