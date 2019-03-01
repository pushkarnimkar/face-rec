#ifndef QUANT_H
#define QUANT_H

#include <stdio.h>
#include <stdint.h>
#include "enums.h"

typedef struct {

} QuantizationTable;

// Initializes quant_tbl using size number of bytes from _buffer
ParseStatus parse_quant_tbl(
    uint8_t** _buffer, 
    size_t size, 
    QuantizationTable* quant_tbl
);

#endif // ifndef QUANT_H
