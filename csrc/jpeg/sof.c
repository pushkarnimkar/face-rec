#include "sof.h"

ParseStatus parse_sof(uint8_t** _buffer, size_t size, FrameHeader* sof0) {
    *_buffer += size;
    return PARSE_SUCCESS;
}
