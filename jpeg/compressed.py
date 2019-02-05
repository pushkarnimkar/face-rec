from collections import defaultdict
from jpeg.markers import (read_soi, read_eoi, QuantizationTable, HuffmanTable,
                          read_marker, FrameHeader, ScanHeader, ScanComponent)
from jpeg.scan import BitReader, decode_dc, decode_ac
from typing import Dict

import numpy as np


def read_comp(reader, state_dc: dict, component: ScanComponent):
    dc_diff, comp_id = decode_dc(reader, component), component.component_id
    dc_coef = dc_diff if state_dc[comp_id] is None else \
        state_dc[comp_id] + dc_diff
    ac_coef = decode_ac(reader, component)
    state_dc[component.component_id] = dc_coef
    return np.insert(ac_coef, 0, dc_coef)


def scan_imcu(reader: BitReader, scan_header: ScanHeader, state_dc: dict):
    block11 = read_comp(reader, state_dc, scan_header.components[1])
    block12 = read_comp(reader, state_dc, scan_header.components[1])
    block20 = read_comp(reader, state_dc, scan_header.components[2])
    block30 = read_comp(reader, state_dc, scan_header.components[3])
    return block11, block12, block20, block30


class Compressed:
    def __init__(self, stream):
        self.quant_tbl: Dict[int, QuantizationTable] = {}
        self.dc_huff_tbl: Dict[int, HuffmanTable] = {}
        self.ac_huff_tbl: Dict[int, HuffmanTable] = {}
        self.app0 = None
        self.sof0 = None
        self.stream = stream
        self.mcus = {}

    def parse(self):
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
                                     self.ac_huff_tbl, self.dc_huff_tbl)

            reader = BitReader(self.stream)
            state_dc, mcus = defaultdict(lambda: None), defaultdict(list)
            for _ in range(2400):
                try:
                    b11, b12, b20, b30 = \
                        scan_imcu(reader, scan_header, state_dc)
                    mcus[1].extend((b11, b12))
                    mcus[2].append(b20)
                    mcus[3].append(b30)
                except Exception as exc:
                    raise exc
                finally:
                    self.mcus = dict(mcus)
            read_eoi(self.stream)
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
