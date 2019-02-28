#include "compressed.h"
#ifdef DEVELOPMENT_ENV
# include <stdio.h>
# include <stdlib.h>
#endif

void read_soi(unsigned char** buffer) {
    if (**buffer != 0xFF || *(*buffer + 1) != 0xD8) {
        parse_failure(SOI_NOT_FOUND);
    }
    *buffer += 2;
}

void parse(Compressed* comp, unsigned char* buffer) {
    read_soi(&buffer);
}

void parse_failure(ParseError error) {
    #ifdef DEVELOPMENT_ENV
    switch (error) {
        case SOI_NOT_FOUND: {
            fprintf(stderr, "SOI NOT FOUND\n");
            exit(1);
        }
    }
    #else
    
    #endif
}