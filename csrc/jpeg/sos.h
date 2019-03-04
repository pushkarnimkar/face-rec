#ifndef SOS_H
#define SOS_H

#include <stdio.h>
#include <stdint.h>

#include "enums.h"
#include "quant.h"
#include "huff.h"
#include "sof.h"

typedef struct ScanComponent {
    FrameComponent* fcomp;
    QuantizationTable* quant_tbl;
    HuffmanTable* dc_huff_tbl;
    HuffmanTable* ac_huff_tbl;
} ScanComponent;

typedef struct ScanHeader {
    uint8_t n_comp;
    ScanComponent comp[3];
    FrameHeader* sof0;
} ScanHeader;

// Initializes huff_tbl using size number of bytes from _buffer
ParseStatus 
parse_sos(
    uint8_t** _buffer, size_t size, ScanHeader* sos, 
    FrameHeader* sof0, QuantizationTable quant_tbl[], 
    HuffmanTable dc_huff_tbl[], HuffmanTable ac_huff_tbl[]
);

#endif // ifndef SOS_H
