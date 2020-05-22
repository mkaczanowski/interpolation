#include <stdint.h>
#include "lib.h"

CUDA_DEV void bilinearTransform(
	int bx,
	int by,
	uint8_t *output,
	uint8_t *input,
	uint16_t pitchOutput,
	uint16_t pitchInput,
	uint8_t bytesPerPixelInput,
	uint8_t bytesPerPixelOutput,
	float xRatio,
	float yRatio
){
    #ifdef __CUDACC__
	bx = blockIdx.x;
	by = blockIdx.y;
	#endif

    int x = (int) (xRatio * bx);
    int y = (int) (yRatio * by);

    uint8_t *a; uint8_t *b; uint8_t *c; uint8_t *d;
    float xDist, yDist, blue, red, green;

    // X and Y distance difference
    xDist = (xRatio * bx) - x;
    yDist = (yRatio * by) - y;

    // Points
    a = input + y * pitchInput + x * bytesPerPixelInput;
    b = input + y * pitchInput + (x+1) * bytesPerPixelInput;
    c = input + (y+1) * pitchInput + x * bytesPerPixelInput;
    d = input + (y+1) * pitchInput + (x+1) * bytesPerPixelInput;

    // blue
    blue = (a[2])*(1 - xDist)*(1 - yDist) + (b[2])*(xDist)*(1 - yDist) + (c[2])*(yDist)*(1 - xDist) + (d[2])*(xDist * yDist);

    // green
    green = ((a[1]))*(1 - xDist)*(1 - yDist) + (b[1])*(xDist)*(1 - yDist) + (c[1])*(yDist)*(1 - xDist) + (d[1])*(xDist * yDist);

    // red
    red = (a[0])*(1 - xDist)*(1 - yDist) + (b[0])*(xDist)*(1 - yDist) + (c[0])*(yDist)*(1 - xDist) + (d[0])*(xDist * yDist);

    uint8_t *p = output + by * pitchOutput + bx * bytesPerPixelOutput;
    *(uint32_t*)p = 0xff000000 | ((((int)red) << 16)) | ((((int)green) << 8)) | ((int)blue);
}
