#include "sos.h"

ParseStatus parse_sos(uint8_t** _buffer, size_t size, SegmentHeader* sos) {
    *_buffer += size;
    return PARSE_SUCCESS;
}
