#ifndef COEF_H
#define COEF_H

#include <stdint.h>
#include "../config.h"
#include "enums.h"
#include "sos.h"

#define INIT_BITREADER \
    BitReader reader = {buffer, 0, -1};\
    ParseStatus status = refill_bits_reader(&reader); \

#define DECODE_AC \
    rs = decode(reader, huff_tbl); \
    r = rs >> 4, s = rs & 0x0F; \
    k++;

#define PEEK_SIZE 8
#define COEF_DTYPE int16_t

#ifdef IMCUPTR_SAVE
typedef struct ImcuPtr {
    // pointer to base location in the reader
    uint8_t* buffer;
    // pointer to bit offset from base
    uint8_t offset;
} ImcuPtr;
#endif

typedef struct BitReader {
    uint8_t** stream;
    uint32_t bits;
    int8_t bits_ptr;
} BitReader;

ParseStatus read_imcu_coef(
    BitReader* reader, ScanHeader sos, 
    COEF_DTYPE* prev_dc, COEF_DTYPE* coef
);

ParseStatus refill_bits_reader(BitReader* reader);

#endif // ifndef COEF_H
