#include "../config.h"

typedef struct {
    unsigned char parsed;
} Compressed;

typedef enum {
    SOI_NOT_FOUND
} ParseError;

void parse(Compressed* comp, unsigned char* buffer);

void parse_failure(ParseError);

