#include "scan.h"
#include <string.h>

#define COEF_BUFFER_SIZE 256
#define COEF_NUMBER_COMP 3

ParseStatus 
scan(uint8_t** buffer, ScanHeader sos, ScanOutput* out) {
    INIT_BITREADER
    COEF_DTYPE prev_dc[COEF_NUMBER_COMP];
    memset((void*) prev_dc, 0, COEF_NUMBER_COMP * sizeof(COEF_DTYPE));
    COEF_DTYPE coef[COEF_BUFFER_SIZE];

    int imcu_hori_count = sos.sof0->y / sos.sof0->imcu_wd / 8;
    int imcu_vert_count = sos.sof0->x / sos.sof0->imcu_ht / 8;
    int imcu_net_count = imcu_hori_count * imcu_vert_count;

    uint8_t mcu[64];
    for (int i = 0; i < imcu_net_count; i++) {
        memset((void*) coef, 0, COEF_BUFFER_SIZE * sizeof(COEF_DTYPE));
        status = read_imcu_coef(&reader, sos, prev_dc, coef);

        // memset((void*) mcu, 0, 64 * sizeof(COEF_DTYPE));
        idct2(coef, mcu);

        if (status != PARSE_SUCCESS) {
            return status;
        }
    }

    return PARSE_SUCCESS;
}
