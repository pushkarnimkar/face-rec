#include "sos.h"

void parse_scan_comp(
    uint8_t** buffer, ScanComponent* comp, 
    FrameHeader* sof0, QuantizationTable quant_tbl[], 
    HuffmanTable dc_huff_tbl[], HuffmanTable ac_huff_tbl[]
) {
    uint8_t comp_id = *((*buffer)++);
    comp->fcomp = &sof0->comp[comp_id];

    uint8_t td = (**buffer & 0xF0) >> 4;
    uint8_t ta = *((*buffer)++) & 0x0F;
    comp->dc_huff_tbl = dc_huff_tbl + td;
    comp->ac_huff_tbl = ac_huff_tbl + ta;
    comp->quant_tbl = quant_tbl + comp->fcomp->quant_tid;
}

ParseStatus parse_sos(
    uint8_t** buffer, size_t size, ScanHeader* sos, 
    FrameHeader* sof0, QuantizationTable quant_tbl[], 
    HuffmanTable dc_huff_tbl[], HuffmanTable ac_huff_tbl[]
) {
    uint8_t* __init_buffer = *buffer;
    sos->n_comp = *((*buffer)++);
    for (int i = 0; i < sos->n_comp; i++) {
        parse_scan_comp(
            buffer, &sos->comp[i], sof0, 
            quant_tbl, dc_huff_tbl, ac_huff_tbl
        );
    }
    *buffer += 3;
    sos->sof0 = sof0;

    if (*buffer == __init_buffer + size) {
        return PARSE_SUCCESS;
    } else {
        return HUFFMAN_TABLE_PARSE_ERROR;
    }
}
