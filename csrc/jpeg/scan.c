#include "scan.h"
#include "../maths/common.h"
#include <string.h>

#define COEF_BUFFER_SIZE 256
#define COEF_NUMBER_COMP 3
#define FACTOR 4
#define WIDTH 2

ParseStatus 
scan(uint8_t** buffer, ScanHeader sos, ScanOutput* out) {
    INIT_BITREADER
    COEF_DTYPE prev_dc[COEF_NUMBER_COMP];
    memset((void*) prev_dc, 0, COEF_NUMBER_COMP * sizeof(COEF_DTYPE));
    COEF_DTYPE coef[COEF_BUFFER_SIZE];

    int imcu_hori_count = sos.sof0->y / sos.sof0->imcu_wd / 8;
    int imcu_vert_count = sos.sof0->x / sos.sof0->imcu_ht / 8;
    int imcu_net_count = imcu_hori_count * imcu_vert_count;

    float mcu[64];
    for (int imcu = 0; imcu < imcu_net_count; imcu++) {
        memset((void*) coef, 0, COEF_BUFFER_SIZE * sizeof(COEF_DTYPE));
        status = read_imcu_coef(&reader, sos, prev_dc, coef);

        // memset((void*) mcu, 0, 64 * sizeof(COEF_DTYPE));
        idct2(coef, mcu);
        for (int j = 0; j < 8; j += FACTOR) {
            for (int i = 0; i < 8; i += FACTOR) {
                int mcu_x = (imcu * 2) / 80, mcu_y = (imcu * 2) % 80;
                int offset = (mcu_x * 80 * WIDTH  + mcu_y) * WIDTH;
                float* out_ptr = out->image + offset;
                // because we increment in steps of factor
                offset = (j * 160 + i) / FACTOR;
                out_ptr += offset;
                for (int y = j; y < j + FACTOR; y++) {
                    for (int x = i; x < i + FACTOR; x++ ) {
                        *out_ptr += mcu[y * 8 + x];
                    }
                }
                *out_ptr /= (FACTOR * FACTOR);
            }
        }
        idct2(coef + 64, mcu);
        for (int j = 0; j < 8; j += FACTOR) {
            for (int i = 0; i < 8; i += FACTOR) {
                int mcu_x = (imcu * 2 + 1) / 80, mcu_y = (imcu * 2 + 1) % 80;
                int offset = (mcu_x * 80 * WIDTH  + mcu_y) * WIDTH;
                float* out_ptr = out->image + offset;
                // because we increment in steps of factor
                offset = (j * 160 + i) / FACTOR;
                out_ptr += offset;
                for (int y = j; y < j + FACTOR; y++) {
                    for (int x = i; x < i + FACTOR; x++ ) {
                        *out_ptr += mcu[y * 8 + x];
                    }
                }
                *out_ptr /= (FACTOR * FACTOR);
            }
        }

        if (status != PARSE_SUCCESS) {
            return status;
        }
    }

    return PARSE_SUCCESS;
}
