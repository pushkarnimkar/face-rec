#ifndef COMPRESSED_H
#define COMPRESSED_H

#include <stdint.h>
#include "../config.h"
#include "enums.h"
#include "huff.h"
#include "quant.h"
#include "sof0.h"
#include "sos.h"

typedef struct {
    unsigned char is_parsed;
    HuffmanTable huff_tbl[4];
    QuantizationTable quant_tbl[2];
    FrameHeader sof0;
    SegmentHeader sos;
} Compressed;

void parse(Compressed* comp, uint8_t* buffer);

void parse_failure(ParseStatus);

#endif // ifndef COMPRESSED_H
