#ifndef ENUMS_H
#define ENUMS_H

typedef enum ParseStatus {
    SOI_NOT_FOUND,
    HUFFMAN_TABLE_PARSE_ERROR,
    PARSE_SUCCESS,
    SCAN_FAILURE,
    REACHED_EOI,
    UNKNOWN_MARKER
} ParseStatus;

typedef enum MarkerType {
    SOI = 0xD8,
    APP0 = 0xE0,
    DQT = 0xDB,
    DHT = 0xC4,
    SOF0 = 0xC0,
    SOS = 0xDA,
    EOI = 0xD9
} MarkerType;

#endif // ifndef ENUMS_H
