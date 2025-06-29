#ifndef COMPRESSED_H
#define COMPRESSED_H

#include <stdint.h>
#include "../config.h"
#include "scan.h"

typedef struct Compressed {
    unsigned char is_parsed;
    QuantizationTable quant_tbl[2];
    HuffmanTable dc_huff_tbl[2];
    HuffmanTable ac_huff_tbl[2];
    FrameHeader sof0;
    ScanHeader sos;
    ScanOutput scanned;
} Compressed;

void parse(Compressed* comp, uint8_t* buffer);

void parse_failure(ParseStatus);

#endif // ifndef COMPRESSED_H
