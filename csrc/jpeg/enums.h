#ifndef ENUMS_H
#define ENUMS_H

typedef enum {
    SOI_NOT_FOUND,
    PARSE_SUCCESS,
    UNKNOWN_MARKER
} ParseStatus;

typedef enum {
    SOI = 0xD8,
    APP0 = 0xE0,
    DQT = 0xDB,
    DHT = 0xC4,
    SOF0 = 0xC0,
    SOS = 0xDA,
    EOI = 0xD9
} MarkerType;

#endif // ifndef ENUMS_H