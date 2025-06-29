from typing import Tuple, Dict, List
from recon.jpeg.common import ByteReader
import numpy as np


class JPEGParseError(Exception):
    pass


class SOINotFoundError(JPEGParseError):
    pass


class UndefinedMarkerError(JPEGParseError):
    def __init__(self, mid: bytes):
        super(JPEGParseError, self).__init__()
        self.args = ["Undefined marker", "0x" + mid.hex()]


MARKERS = {
    b"\xD8": "SOI",
    b"\xE0": "APP0",
    b"\xDB": "DQT",
    b"\xC0": "SOF0",
    b"\xC4": "DHT",
    b"\xDA": "SOS",
    b"\xD9": "EOI"
}


def read_marker(stream: ByteReader) -> Tuple[str, bytes]:
    try:
        while True:
            byte = stream.read(1)
            if byte != b"\xFF":
                continue
            next_byte = stream.read(1)
            if next_byte == b"\x00":
                continue
            marker = MARKERS[next_byte]
            break
        size = int.from_bytes(stream.read(2), byteorder="big")
        return marker, stream.read(size - 2) if size != 0 else (marker, b"")
    except KeyError as ke:
        raise UndefinedMarkerError(ke.args[0])


def read_soi(stream: ByteReader) -> None:
    while stream.read(1) != b"\xFF":
        continue
    if stream.read(1) != b"\xD8":
        raise SOINotFoundError
    return


def read_eoi(stream: ByteReader) -> None:
    while stream.read(1) != b"\xFF":
        continue
    if stream.read(1) != b"\xD9":
        raise SOINotFoundError
    return


def parse_byte_params(b: int) -> tuple:
    return (b & 0xf0) >> 4, b & 0x0f


class QuantizationTable:
    order = np.array([[ 0,  1,  5,  6, 14, 15, 27, 28],
                      [ 2,  4,  7, 13, 16, 26, 29, 42],
                      [ 3,  8, 12, 17, 25, 30, 41, 43],
                      [ 9, 11, 18, 24, 31, 40, 44, 53],
                      [10, 19, 23, 32, 39, 45, 52, 54],
                      [20, 22, 33, 38, 46, 51, 55, 60],
                      [21, 34, 37, 47, 50, 56, 59, 61],
                      [35, 36, 48, 49, 57, 58, 62, 63]], dtype=np.int8)

    def __init__(self, bs: bytes):
        self.table = np.frombuffer(bs[1:], np.uint8)[self.order]
        self.pq, self.table_id = parse_byte_params(bs[0])


def _make_huff_size(bits: List[int]) -> Tuple[np.ndarray, int]:
    total, k = sum(bits), 0
    huff_size = np.ndarray((total + 1,), dtype=np.uint8)

    for length_m1, count in enumerate(bits):
        for _ in range(count):
            huff_size[k] = length_m1 + 1
            k += 1

    huff_size[-1], last_k = 0, k
    return huff_size, last_k


def _make_huff_code(huff_size: np.ndarray) -> np.ndarray:
    k, code, si, huff_code = 0, 0, huff_size[0], []
    while huff_size[k] != 0:
        while huff_size[k] == si:
            huff_code.append(code)
            k, code = k + 1, code + 1
        if huff_size[k] == 0:
            return np.array(huff_code, np.uint16)
        while huff_size[k] != si:
            code, si = code << 1, si + 1
    return np.array(huff_code, np.uint16)


class HuffmanTable:
    def __init__(self, bs: bytes):
        self.is_ac, self.table_id = parse_byte_params(bs[0])
        # bits contains size of each bucket of code length
        self.bits = list(bs[1:17])
        # huff_val contains each symbol that is encoded
        self.huff_val = list(bs[17:17 + sum(self.bits)])

        # huff_size contains length of code for symbol at index in huff_val
        self.huff_size, last_k = _make_huff_size(self.bits)
        # huff_code contains code associated with symbol at index in huff_val
        self.huff_code = _make_huff_code(self.huff_size)

        # contain minimum and maximum codes of each size
        self.min_code, self.max_code = (np.repeat(-1, 17).astype(np.uint16),
                                        np.repeat(-1, 17).astype(np.uint16))
        # contains index in huff_val from which codes of size index start
        self.val_ptr = np.zeros(17, dtype=np.uint8)
        self._min_max_codes()

        # lookups enable fast parsing of codes. But if lookup fails,
        # we fallback to our regular scanning process.
        self.lookup_size = np.zeros(256, dtype=np.uint8)
        self.lookup_code = np.zeros(256, dtype=np.uint8)
        self._make_lookup()

    def _min_max_codes(self):
        val_idx = 0
        for i in range(1, 17):
            if self.bits[i - 1] == 0:
                self.max_code[i] = -1
                continue
            self.val_ptr[i] = val_idx
            self.min_code[i] = self.huff_code[val_idx]
            val_idx += (self.bits[i - 1] - 1)
            self.max_code[i] = self.huff_code[val_idx]
            val_idx += 1

    def _make_lookup(self):
        _code_index = 0
        for si in range(1, 9):
            if self.min_code[si] == 0xFF:
                continue
            while self.huff_size[_code_index] == si:
                _rem, _code, val = (8 - si, self.huff_code[_code_index],
                                    self.huff_val[_code_index])
                for _nc in range(1 << _rem):
                    code = (_code << _rem) + _nc
                    self.lookup_size[code] = si
                    self.lookup_code[code] = val
                _code_index += 1

        # while index != 256:
        #     _size = self.huff_size[index]
        #     _remaining = 8 - _size


class FrameHeader:
    class Component:
        def __init__(self, bs: bytes):
            self.component_id = bs[0]
            self.hsf, self.vsf = parse_byte_params(bs[1])
            self.quant_tbl = bs[2]

    def __init__(self, bs: bytes):
        self.precision = bs[0]
        self.y, self.x = (int.from_bytes(bs[1:3], byteorder="big"),
                          int.from_bytes(bs[3:5], byteorder="big"))
        nf, rest, self.components = bs[5], bs[6:], {}
        rest = self._parse_components(nf, rest)
        assert len(rest) == 0
        self.imcu_height, self.imcu_width = self._compute_imcu_dims()

    def _parse_components(self, nf: int, rest: bytes):
        for _ in range(nf):
            component, rest = self.Component(rest[:3]), rest[3:]
            self.components[component.component_id] = component
        return rest

    def _compute_imcu_dims(self):
        assert len(self.components) != 0
        _imcu_height, _imcu_width = 0, 0
        for _, component in self.components.items():
            _imcu_height = _imcu_height \
                if _imcu_height >= component.vsf else component.vsf
            _imcu_width = _imcu_width \
                if _imcu_width >= component.hsf else component.hsf
        return _imcu_height, _imcu_width

    def get_component(self, cid: int):
        return self.components[cid]


class ScanComponent:
    def __init__(self, bs: bytes,
                 frame_header: FrameHeader,
                 ac_huff_tbl: Dict[int, HuffmanTable],
                 dc_huff_tbl: Dict[int, HuffmanTable],
                 quant_tbl: Dict[int, QuantizationTable]):

        _component = frame_header.get_component(bs[0])
        self.component_id = _component.component_id
        td, ta = parse_byte_params(bs[1])
        self.dc_huff_tbl = dc_huff_tbl[td]
        self.ac_huff_tbl = ac_huff_tbl[ta]
        self.quant_tbl = quant_tbl[_component.quant_tbl]


class ScanHeader:
    def __init__(self, bs: bytes,
                 frame_header: FrameHeader,
                 ac_huff_tbl: Dict[int, HuffmanTable],
                 dc_huff_tbl: Dict[int, HuffmanTable],
                 quant_tbl: Dict[int, QuantizationTable]):

        ns, rest = bs[0], bs[1:]
        self.components: Dict[int, ScanComponent] = {}
        for _ in range(ns):
            component = ScanComponent(
                rest[:2], frame_header, ac_huff_tbl, dc_huff_tbl, quant_tbl)
            self.components[component.component_id], rest = component, rest[2:]

        self.ss, self.se = rest[0], rest[1]
        self.ah, self.al = parse_byte_params(rest[2])
