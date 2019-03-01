#ifndef SOS_H
#define SOS_H

#include <stdio.h>
#include <stdint.h>
#include "enums.h"

typedef struct {

} SegmentHeader;

// Initializes huff_tbl using size number of bytes from _buffer
ParseStatus parse_sos(uint8_t** _buffer, size_t size, SegmentHeader* header);

#endif // ifndef SOS_H
