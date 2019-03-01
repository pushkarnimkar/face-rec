#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "compressed.h"

void read_soi(unsigned char** _buffer) {
    if (**_buffer != 0xFF || *(*_buffer + 1) != 0xD8) {
        parse_failure(SOI_NOT_FOUND);
    }
    *_buffer += 2;
}

void parse(Compressed* comp, uint8_t* buffer) {
    read_soi(&buffer);
    uint8_t quant_tbl_cnt = 0, huff_tbl_cnt = 0;
    while (1) {
        if (*buffer == 0xFF && *(buffer + 1) != 0x00) {
            size_t size = ((size_t)(*(buffer + 2)) << 8) + (size_t)(*(buffer + 3)) - 2;
            switch (*(buffer + 1)) {
        case APP0: {
            // Skip the marker we dont use APP0
            buffer += size + 4;
            break;
        }
        case DQT: {
            QuantizationTable* tbl = comp->quant_tbl + quant_tbl_cnt++;
            buffer += 4;
            ParseStatus status = parse_quant_tbl(&buffer, size, tbl);
            if (status != PARSE_SUCCESS) {
                parse_failure(status);
            }
            break;
        }
        case DHT: {
            HuffmanTable* tbl = comp->huff_tbl + huff_tbl_cnt++;
            buffer += 4;
            ParseStatus status = parse_huff_tbl(&buffer, size, tbl);
            if (status != PARSE_SUCCESS) {
                parse_failure(status);
            }
            break;
        }
        case SOF0: {
            buffer += 4;
            ParseStatus status = parse_sof(&buffer, size, &comp->sof0);
            if (status != PARSE_SUCCESS) {
                parse_failure(status);
            }
            break;
        }
        case SOS: {
            buffer += 4;
            ParseStatus status = parse_sos(&buffer, size, &comp->sos);
            if (status != PARSE_SUCCESS) {
                parse_failure(status);
            }
            break;
        }
        default: {
            parse_failure(UNKNOWN_MARKER);
        }
            }
        } else {
            while (!(*(buffer) == 0xFF && *(buffer + 1) == EOI)) {
                buffer++;
            }
            break;
        }
    }
}

void parse_failure(ParseStatus status) {
    #ifdef DEVELOPMENT_ENV
    switch (status) {
        case SOI_NOT_FOUND: {
            fprintf(stderr, "SOI NOT FOUND\n");
            exit(1);
        }
        case UNKNOWN_MARKER: {
            fprintf(stderr, "UNKNOWN MARKER\n");
            exit(1);
        }
    }
    #else
    
    #endif
}
