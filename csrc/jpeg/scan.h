#ifndef SCAN_H
#define SCAN_H

#include "coef.h"
#include "../maths/idct.h"

// reason for using structure like this it to give flexibility of 
// adding and removing output components as required
typedef struct ScanOutput {
    #ifdef IMCUPTR_SAVE
    ImcuPtr* imcus;
    #endif
    ParseStatus scan_status;
    float image[19200];
} ScanOutput;

ParseStatus 
scan(uint8_t** buffer, ScanHeader header, ScanOutput* out);

#endif
