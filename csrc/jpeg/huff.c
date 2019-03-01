#include "huff.h"

void make_huff_size(HuffmanTable* huff_tbl, uint8_t* bits) {
    uint8_t k = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < bits[i]; j++) {
            huff_tbl->huff_size[k++] = i + 1;
        }
    }
    huff_tbl->huff_size[k] = (uint8_t)(-1);
}

void make_huff_code(HuffmanTable* huff_tbl) {
    uint8_t* huff_size = huff_tbl->huff_size;
    uint8_t k = 0, si = huff_size[0], end = (uint8_t)(-1);
    uint16_t code = 0;

    while (huff_size[k] != end) {
        while (huff_size[k] == si) {
            huff_tbl->huff_code[k++] = code++;
        }
        if (huff_size[k] == end) {
            break;
        }
        while (huff_size[k] != si) {
            code = code << 1;
            si++;
        }
    }
    huff_tbl->huff_code[k] = (uint16_t)(-1);
}

void make_min_max_codes(HuffmanTable* huff_tbl, uint8_t* bits) {
    uint8_t val_idx = 0;
    for (int i = 1; i < 17; i++) {
        if (bits[i - 1] == 0) {
            huff_tbl->min_code[i] = huff_tbl->max_code[i] = (uint16_t)(-1);
            continue;
        }
        huff_tbl->val_ptr[i] = val_idx;
        huff_tbl->min_code[i] = huff_tbl->huff_code[val_idx];
        val_idx += bits[i - 1] - 1;
        huff_tbl->max_code[i] = huff_tbl->huff_code[val_idx++];
    }
}

void make_lookup_tables(HuffmanTable* huff_tbl) {
    int code_index = 0;
    uint16_t unavailable = (uint16_t)(-1);
    for (int si = 0; si < 9; si++) {
        if (huff_tbl->min_code[si] == unavailable) {
            continue;
        }
        while (huff_tbl->huff_size[code_index] == si) {
            uint8_t rem = 8 - si, sym = huff_tbl->huff_val[code_index];
            uint8_t count = 1 << rem;
            uint16_t code = huff_tbl->huff_code[code_index++];
            for (int j = 0; j < count; j++) {
                uint8_t _code = (code << rem) + j;
                huff_tbl->lookup_code[_code] = sym;
                huff_tbl->lookup_size[_code] = si;
            }
        }
    }
}

ParseStatus 
parse_huff_tbl(uint8_t** __buffer, size_t size, HuffmanTable* huff_tbl) {
    uint8_t* __init_buffer = *__buffer;

    huff_tbl->is_ac = (**__buffer & 0xF0) >> 4;
    huff_tbl->tid = **__buffer & 0x0F;

    uint8_t* bits = ++(*__buffer);
    size_t huff_val_size = 0;
    for (int i = 0; i < 16; i++) {
        huff_val_size += *((*__buffer)++);
    }
    huff_tbl->huff_val = *__buffer;
    *__buffer += huff_val_size;

    make_huff_size(huff_tbl, bits);
    make_huff_code(huff_tbl);
    make_min_max_codes(huff_tbl, bits);
    make_lookup_tables(huff_tbl);

    if (*__buffer == __init_buffer + size) {
        return PARSE_SUCCESS;
    } else {
        return HUFFMAN_TABLE_PARSE_ERROR;
    }
}
