#ifndef HUFF_H
#define HUFF_H

#include <stdio.h>
#include <stdint.h>
#include "enums.h"

typedef struct {

} HuffmanTable;

// Initializes huff_tbl using size number of bytes from _buffer
ParseStatus parse_huff_tbl(
    uint8_t** _buffer, 
    size_t size, 
    HuffmanTable* huff_tbl
);

#endif // ifndef HUFF_H
