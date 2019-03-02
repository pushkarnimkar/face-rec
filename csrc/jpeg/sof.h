#ifndef SOF0_H
#define SOF0_H

#include <stdio.h>
#include <stdint.h>
#include "enums.h"
#include "utils.h"

typedef struct FrameComponent {
    uint8_t comp_id;
    uint8_t hsf;
    uint8_t vsf;
    uint8_t quant_tid;
} FrameComponent;

typedef struct {
    uint8_t precision;
    size_t x;
    size_t y;
    uint8_t n_comp;
    uint8_t imcu_ht;
    uint8_t imcu_wd;
    FrameComponent comp[4];
} FrameHeader;

// Initializes huff_tbl using size number of bytes from _buffer
ParseStatus parse_sof(uint8_t** _buffer, size_t size, FrameHeader* header);

#endif // ifndef SOF0_H
