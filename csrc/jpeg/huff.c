#include "huff.h"

ParseStatus parse_huff_tbl(
    uint8_t** _buffer, 
    size_t size, 
    HuffmanTable* quant_tbl
) {
    *_buffer += size;
    return PARSE_SUCCESS;
}
