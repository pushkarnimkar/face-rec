#ifndef HUFF_H
#define HUFF_H

#include <stdio.h>
#include <stdint.h>
#include "enums.h"

typedef struct HuffmanTable {
    // huff_val contains ordered symbols that are encoded
    uint8_t* huff_val;
    // contains size of each of huff_val (can be dynamically allocated)
    uint8_t huff_size[256];
    // contains code for each of huff_val in same order
    uint16_t huff_code[256];
    // contains index in huff_val from which codes of size equal to index 
    uint8_t val_ptr[17];
    // contain minimum codes of each size equal to index 
    uint16_t min_code[17];
    // contain maximum codes of each size equal to index 
    uint16_t max_code[17];
    // lookups enable fast parsing of codes. But if lookup fails,
    // we fallback to our regular scanning process.
    uint8_t lookup_code[256];
    uint8_t lookup_size[256];
} HuffmanTable;

// Initializes huff_tbl using size number of bytes from _buffer
ParseStatus parse_huff_tbl(
    uint8_t** _buffer, 
    size_t size, 
    HuffmanTable* huff_tbl
);

#endif // ifndef HUFF_H
