#include "quant.h"

ParseStatus parse_quant_tbl(
    uint8_t** _buffer, 
    size_t size, 
    QuantizationTable* quant_tbl
) {
    *_buffer += size;
    return PARSE_SUCCESS;
}
