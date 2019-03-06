#include "sof.h"
#include "utils.h"

void parse_frame_comp(uint8_t** buffer, FrameHeader* sof0) {
    uint8_t comp_id = *((*buffer)++);
    FrameComponent *fcomp = &sof0->comp[comp_id];
    fcomp->comp_id = comp_id;
    fcomp->hsf = ((**buffer) & 0xF0) >> 4;
    fcomp->vsf = *((*buffer)++) & 0x0F;
    fcomp->quant_tid = *((*buffer)++);
}

void compute_imcu_dims(FrameHeader* sof0) {
    sof0->imcu_ht = 0;
    sof0->imcu_wd = 0;
    for (int i = 1; i <= sof0->n_comp; i++) {
        sof0->imcu_ht = MAX(sof0->imcu_ht, sof0->comp[i].vsf);
        sof0->imcu_wd = MAX(sof0->imcu_wd, sof0->comp[i].hsf);
    }
}

ParseStatus 
parse_sof(uint8_t** buffer, size_t size, FrameHeader* sof0) {
    uint8_t* __init_buffer = *buffer;
    sof0->precision = *((*buffer)++);
    sof0->x = READSIZE2B((*buffer));
    sof0->y = READSIZE2B((*buffer));
    sof0->n_comp = *((*buffer)++);

    for (int i = 0; i < sof0->n_comp; i++) {
        parse_frame_comp(buffer, sof0);
    }
    compute_imcu_dims(sof0);

    if (*buffer == __init_buffer + size) {
        return PARSE_SUCCESS;
    } else {
        return HUFFMAN_TABLE_PARSE_ERROR;
    }
}
