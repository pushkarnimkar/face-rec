#ifndef COMPRESSED_H
#define COMPRESSED_H

#include <stdint.h>
#include "../config.h"
#include "enums.h"
#include "huff.h"
#include "quant.h"
#include "sof.h"
#include "sos.h"

typedef struct Compressed {
    unsigned char is_parsed;
    QuantizationTable quant_tbl[2];
    HuffmanTable dc_huff_tbl[2];
    HuffmanTable ac_huff_tbl[2];
    FrameHeader sof0;
    ScanHeader sos;
} Compressed;

void parse(Compressed* comp, uint8_t* buffer);

void parse_failure(ParseStatus);

#endif // ifndef COMPRESSED_H
