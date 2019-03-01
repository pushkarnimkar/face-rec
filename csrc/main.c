#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "config.h"
#include "jpeg/compressed.h"


int main(int argc, char* argv[]) {
    uint8_t buffer[102400];
    #ifdef DEVELOPMENT_ENV
    if (argc == 1) {
        fprintf(stderr, "insufficient arguments\n");
        exit(1);
    }
    FILE* fp = fopen(argv[1], "rb");
    int count = fread((void*)buffer, 1, 102400, fp);
    printf("read %d bytes\n", count);
    fclose(fp);
    #endif
    Compressed comp;
    parse(&comp, buffer);
    return 0;
}
