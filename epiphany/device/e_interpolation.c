#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "e_lib.h"
#include "../../lib/lib.h"

// core coords to relative position lookup table
int cores[4][4] = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12},
    {13, 14, 15, 16}
};

const char old_shm_name[] = "old_pixels_shm"; 
const char new_shm_name[] = "new_pixels_shm"; 

e_memseg_t old_emem;
e_memseg_t new_emem;

void waitForLock(volatile int* debug, volatile int* lock) {
    *debug = 0;
    *lock = 1;

    while(1) {
        if (*lock == 1) {
            continue;
        }
        return;
    }
}

int main(void) {
	volatile int* lock = (int*) 0x7032;
	volatile int* debug = (int*) 0x7036;

    volatile uint16_t* pitchOutput = (uint16_t*) 0x7000;
    volatile uint16_t* pitchInput = (uint16_t*) 0x7004;
    volatile uint16_t* bytesPerPixelInput = (uint16_t*) 0x7008;
    volatile uint16_t* bytesPerPixelOutput = (uint16_t*) 0x7012;
    volatile int* newWidth = (int*) 0x7016;
    volatile int* newHeight = (int*) 0x7020;
    volatile float* xRatio = (float*) 0x7024;
    volatile float* yRatio = (float*) 0x7028;

    waitForLock(debug, lock);

    int row = e_group_config.core_row;
    int col = e_group_config.core_col;
    int total_cores = e_group_config.group_rows * e_group_config.group_cols;

    int core_num = cores[row][col];

    if ( E_OK != e_shm_attach(&old_emem, old_shm_name) ) {
        return EXIT_FAILURE;
    }

    if ( E_OK != e_shm_attach(&new_emem, new_shm_name) ) {
        return EXIT_FAILURE;
    }

    e_memseg_t *oldPixelsPmem = (e_memseg_t*) &old_emem;
    uint8_t* oldPixels = (uint8_t *) (oldPixelsPmem->ephy_base);

    e_memseg_t *newPixelsPmem = (e_memseg_t*) &new_emem;
    uint8_t* newPixels = (uint8_t *) (newPixelsPmem->ephy_base);

    // split input image between cores
    int x_chunk_size = (int) (*newWidth / total_cores);
    int start = (core_num - 1) * x_chunk_size;
    int end = start + x_chunk_size;
    if (core_num == 16) {
        end = *newWidth; // the last core handles remaining chunk
    }

    for (int x = start; x < end; x++) {
        for (int y = 0; y < *newHeight; y++) {
            bilinearTransform(
                x,
                y,
                newPixels,
                oldPixels,

                *pitchOutput,
                *pitchInput,
                *bytesPerPixelInput,
                *bytesPerPixelOutput,
                *xRatio,
                *yRatio
            );
        }
    }

    *debug = total_cores;
    *lock = 1;

    return EXIT_SUCCESS;
}
