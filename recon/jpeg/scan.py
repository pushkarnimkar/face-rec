from recon.jpeg.common import BitReader
from recon.jpeg.markers import HuffmanTable, ScanComponent

import numpy as np


def decode(reader: BitReader, huff_tbl: HuffmanTable):
    code, i = reader.next_bit(), 1
    while code > huff_tbl.max_code[i] or huff_tbl.max_code[i] == 0xFFFF:
        i, code = i + 1, (code << 1) + reader.next_bit()
    val_idx = huff_tbl.val_ptr[i] + code - huff_tbl.min_code[i]
    value = huff_tbl.huff_val[val_idx]
    return value


def receive(reader: BitReader, ssss: int):
    i, v = 0, 0
    while i != ssss:
        i, v = i + 1, (v << 1) + reader.next_bit()
    return v, i


def ones(s: int):
    if s == 0:
        return 0
    v, i = 1, 1
    while i != s:
        v, i = (v << 1) + 1, i + 1
    return v


def read_twos_comp(v: int, i: int):
    """read twos complement of value v coded with i bits"""
    return -((v ^ ones(v.bit_length())) + 1) \
        if (v >> (i - 1)) & 1 == 1 else v


def extend(v, t):
    vt = 1 << (t - 1)
    return v + ((-1 << t) + 1) if v < vt else v


def decode_dc(reader: BitReader, scan_component: ScanComponent):
    t = decode(reader, scan_component.dc_huff_tbl)
    if t == 0:
        return t
    diff, _ = receive(reader, t)
    return extend(diff, t)


def _decode_ac(reader: BitReader, huff_tbl: HuffmanTable):
    rs = decode(reader, huff_tbl)
    return rs >> 4, rs % 16


def _decode_zz(reader: BitReader, ssss: int):
    zz_k, _ = receive(reader, ssss)
    return extend(zz_k, ssss)


def decode_ac(reader: BitReader, scan_component: ScanComponent):
    k, zz, tbl = (0, np.zeros((63,), dtype=np.int16),
                  scan_component.ac_huff_tbl)
    rrrr, ssss = _decode_ac(reader, tbl)
    while k != 63:
        if ssss != 0:
            k += rrrr
            zz[k] = _decode_zz(reader, ssss)
        elif ssss == 0 and rrrr == 15:
            k = k + 16
        elif ssss == 0 and rrrr == 0:
            break
        (rrrr, ssss), k = _decode_ac(reader, tbl), k + 1
    return zz
