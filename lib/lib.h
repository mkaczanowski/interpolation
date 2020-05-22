#include <stdint.h>

#ifdef __CUDACC__
#define CUDA_DEV __device__
#else
#define CUDA_DEV
#endif

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
);
