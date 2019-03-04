#include "scan.h"

ParseStatus refill_bits_reader(BitReader* reader) {
    uint8_t read_bytes = (32 - (uint8_t)(reader->bits_ptr + 1)) / 8, next;
    while (read_bytes--) {
        next = *((*reader->stream)++);
        if (next == 0xFF) {
            uint8_t tmp = *((*reader->stream)++);
            if (tmp != 0) {
                return REACHED_EOI;
            }
        }
        reader->bits = (reader->bits << 8) + next;
        reader->bits_ptr += 8;
    }
    return PARSE_SUCCESS;
}

// reads next bit while making required changes to reader state
uint8_t read_bit(BitReader* reader) {
    if (reader->bits_ptr == -1) {
        refill_bits_reader(reader);
    }
    return (reader->bits >> reader->bits_ptr--) & 0x01;
}

// peeks one byte from bit reader without removing from buffer.
// Makes required changes to reader state.
uint8_t peek(BitReader* reader) {
    if (reader->bits_ptr < PEEK_SIZE) {
        refill_bits_reader(reader);
    }
    uint32_t mask = (1 << PEEK_SIZE) - 1;
    uint8_t shift_bits = reader->bits_ptr + 1 - PEEK_SIZE;
    return (uint8_t)((reader->bits >> shift_bits) & mask);
}

void seek(BitReader* reader, uint8_t count) {
    reader->bits_ptr -= count;
}

uint8_t decode(BitReader* reader, HuffmanTable* huff_tbl) {
    uint8_t value = peek(reader), size;
    if ((size = huff_tbl->lookup_size[value]) != 0) {
        seek(reader, size);
        return huff_tbl->lookup_code[value];
    } else {
        uint16_t code = 0;
        for (size = 1; size <= 16; size++) {
            code = (code << 1) + read_bit(reader);
            uint8_t max_code = huff_tbl->max_code[size];
            if (code <= max_code && max_code != 0xFFFF) {
                break;
            }
        }
        uint8_t index = huff_tbl->val_ptr[size] + 
            (code - huff_tbl->min_code[size]);
        return *(huff_tbl->huff_val + index);
    }
}

uint8_t receive(BitReader* reader, uint8_t s) {
    uint8_t i = 0, v = 0;
    while (i++ != s) {
        v = (v << 1) + read_bit(reader);
    }
    return v;
}

COEF_DTYPE extend(uint8_t v, uint8_t t) {
    COEF_DTYPE vt = 1 << (t -1);
    return v < vt ? (COEF_DTYPE) v + (-1 << t) + 1 : v;
}

ParseStatus read_dc_coef(
    BitReader* reader, HuffmanTable* huff_tbl, QuantizationTable* quant_tbl, 
    COEF_DTYPE* coef, COEF_DTYPE* prev_dc
) {
    uint8_t t = decode(reader, huff_tbl);
    if (t == 0) {
        *coef = t;
        return PARSE_SUCCESS;
    }
    COEF_DTYPE diff = extend(receive(reader, t), t);
    *prev_dc += diff * quant_tbl->quant_mat[0];
    *coef = *prev_dc;
    return PARSE_SUCCESS;
}

ParseStatus read_ac_coef(
    BitReader* reader, HuffmanTable* huff_tbl, 
    QuantizationTable* quant_tbl, COEF_DTYPE* coef
) {
    uint8_t rs, r, s, k = 0;
    DECODE_AC
    while (k != 64) {
        if (s != 0 || (s == 0 && r == 0x0F)) {
            k = k + r;
            if (s != 0) {
                COEF_DTYPE _coef = extend(receive(reader, s), s);
                coef[ZZ_INV[k]] = _coef * quant_tbl->quant_mat[ZZ_INV[k]];
            }
        } else if (r == 0x00) {
            break;
        }
        DECODE_AC
    }
    return PARSE_SUCCESS;
}

ParseStatus read_imcu_coef(
    BitReader* reader, ScanHeader sos, 
    COEF_DTYPE* prev_dc, COEF_DTYPE* coef
) {
    uint8_t mcu_idx = 0;
    for (int comp_idx = 0; comp_idx < sos.n_comp; comp_idx++) {
    ScanComponent comp = sos.comp[comp_idx];
    for (int _y = 0; _y < comp.fcomp->vsf; _y++) {
        for (int _x = 0; _x < comp.fcomp->hsf; _x++) {
            COEF_DTYPE* mcu_coef = coef + mcu_idx++ * 64;
            ParseStatus status = read_dc_coef(
                reader, comp.dc_huff_tbl, comp.quant_tbl, 
                mcu_coef, prev_dc + comp_idx
            );
            if (status != PARSE_SUCCESS) {
                return status;
            }
            status = read_ac_coef(
                reader, comp.ac_huff_tbl, comp.quant_tbl, mcu_coef
            );
            if (status != PARSE_SUCCESS) {
                return status;
            }
        }
    }
    }
    return PARSE_SUCCESS;
}

ParseStatus 
scan(uint8_t** buffer, ScanHeader sos, ScanOutput* out) {
    INIT_BITREADER
    COEF_DTYPE prev_dc[10];
    COEF_DTYPE coef[1024];

    status = read_imcu_coef(&reader, sos, prev_dc, coef);
    return PARSE_SUCCESS;
}
