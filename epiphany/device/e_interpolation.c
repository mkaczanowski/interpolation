#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "e_lib.h"
#include "../../lib/lib.h"

volatile int *service_id = (int *)0x7000;
volatile int *execution_counter = (int *)0x7020;

const char old_shm_name[] = "old_pixels_shm"; 
const char new_shm_name[] = "new_pixels_shm"; 

e_memseg_t old_emem;
e_memseg_t new_emem;

volatile uint16_t* pitchOutput = (uint16_t*) 0x7000;
volatile uint16_t* pitchInput = (uint16_t*) 0x7008;
volatile uint16_t* bytesPerPixelInput = (uint16_t*) 0x7016;
volatile uint16_t* bytesPerPixelOutput = (uint16_t*) 0x7024;
volatile float* xRatio = (float*) 0x7032;
volatile float* yRatio = (float*) 0x7040;
volatile int* imageByteLength = (int*) 0x7048;
volatile int* newImageByteLength = (int*) 0x7056;
volatile int* newWidth = (int*) 0x7064;
volatile int* newHeight = (int*) 0x7072;



volatile int* total = (int*) 0x7080;

int main(void) {
    *service_id = -1;
    *execution_counter = 0;

    if ( E_OK != e_shm_attach(&old_emem, old_shm_name) ) {
        return EXIT_FAILURE;
    }

    if ( E_OK != e_shm_attach(&new_emem, new_shm_name) ) {
        return EXIT_FAILURE;
    }

	uint8_t *oldPixels;
    e_read((void*) &old_emem, oldPixels, 0, 0, 0, *imageByteLength);

	uint8_t *newPixels;
    e_read((void*) &new_emem, newPixels, 0, 0, 0, *newImageByteLength);

	*total = 0;

    for (int x = 0; x < *newWidth; x++) {
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
			*total = *total +1 ;
        }
    }

    return EXIT_SUCCESS;
}
