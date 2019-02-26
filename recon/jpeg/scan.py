from recon.jpeg.common import BitReader
from recon.jpeg.markers import HuffmanTable, ScanComponent

import numpy as np


def decode(reader: BitReader, huff_tbl: HuffmanTable):
    _peek = reader.peek()
    if huff_tbl.lookup_size[_peek] != 0:
        reader.seek(huff_tbl.lookup_size[_peek])
        return huff_tbl.lookup_code[_peek]

    code, si = reader.next_bit(), 1
    while code > huff_tbl.max_code[si] or huff_tbl.max_code[si] == 0xFFFF:
        si, code = si + 1, (code << 1) + reader.next_bit()
    val_idx = huff_tbl.val_ptr[si] + code - huff_tbl.min_code[si]
    value = huff_tbl.huff_val[val_idx]
    return value


def receive(reader: BitReader, ssss: int):
    i, v = 0, 0
    while i != ssss:
        i, v = i + 1, (v << 1) + reader.next_bit()
    return v, i


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
