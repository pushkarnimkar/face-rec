#ifndef SOF0_H
#define SOF0_H

#include <stdio.h>
#include <stdint.h>
#include "enums.h"

typedef struct {

} FrameHeader;

// Initializes huff_tbl using size number of bytes from _buffer
ParseStatus parse_sof(uint8_t** _buffer, size_t size, FrameHeader* header);

#endif // ifndef SOF0_H
