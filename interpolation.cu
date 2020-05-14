#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "png/savepng.h"

__global__ void cudaTransform(uint8_t *output, uint8_t *input, uint16_t pitchOutput, uint16_t pitchInput, uint16_t bytesPerPixelInput, uint16_t bytesPerPixelOutput, float xRatio, float yRatio){
        int x = (int) (xRatio * blockIdx.x);
        int y = (int) (yRatio * blockIdx.y);

        uint8_t *a; uint8_t *b; uint8_t *c; uint8_t *d;
        float xDist, yDist, blue, red, green;

        // X and Y distance difference
        xDist = (xRatio * blockIdx.x) - x;
        yDist = (yRatio * blockIdx.y) - y;

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

        uint8_t *p = output + blockIdx.y * pitchOutput + blockIdx.x * bytesPerPixelOutput;
        *(uint32_t*)p = 0xff000000 | ((((int)red) << 16)) | ((((int)green) << 8)) | ((int)blue);
}


void cudasafe(int error, const char* message, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s : %i. In %s line %d\n", message, error, file, line);
        exit(-1);
    }
}

int main(void) {
    uint32_t amask = 0xff000000;
    uint32_t rmask = 0x00ff0000;
    uint32_t gmask = 0x0000ff00;
    uint32_t bmask = 0x000000ff;

    SDL_Surface *image = IMG_Load ("test.jpg");
    int imageByteLength = image->w * image->h * sizeof(uint8_t)*image->format->BytesPerPixel;

    if (!image){
          printf ( "IMG_Load: %s\n", IMG_GetError () );
          return 1;
    }

    // New width of image
    int rWidth = 3000;
    int newWidth = image->w + (rWidth-image->w);
    int newHeight = image->h + (rWidth-image->w);
    dim3 grid(newWidth,newHeight);

    // Create scaled image surface
    SDL_Surface *newImage = SDL_CreateRGBSurface(SDL_SWSURFACE, newWidth, newHeight, 32, rmask, gmask, bmask, amask);
    int newImageByteLength = newImage->w * newImage->h * sizeof(uint8_t)*newImage->format->BytesPerPixel;

    float xRatio = ((float)(image->w-1))/newImage->w;
    float yRatio = ((float)(image->h-1))/newImage->h;

    // Create pointer to device and host pixels
    uint8_t *pixels = (uint8_t*)image->pixels;
    uint8_t *pixels_dyn;

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy original image
    cudasafe(cudaMalloc((void **) &pixels_dyn, imageByteLength),"Original image allocation ",__FILE__,__LINE__);
    cudasafe(cudaMemcpy(pixels_dyn, pixels,  imageByteLength, cudaMemcpyHostToDevice),"Copy original image to device ",__FILE__,__LINE__);

    // Allocate new image on DEVICE
    uint8_t *newPixels_dyn;
    uint8_t *newPixels = (uint8_t*)malloc(newImageByteLength);

    cudasafe(cudaMalloc((void **) &newPixels_dyn, newImageByteLength),"New image allocation ",__FILE__,__LINE__);

    // Start measuring time
    cudaEventRecord(start, 0);

    // Do the bilinear transform on CUDA device
    cudaTransform<<<grid,1>>>(newPixels_dyn, pixels_dyn, newImage->pitch, image->pitch, image->format->BytesPerPixel, newImage->format->BytesPerPixel, xRatio, yRatio);

    // Stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Copy scaled image to host
    cudasafe(cudaMemcpy(newPixels, newPixels_dyn, newImageByteLength, cudaMemcpyDeviceToHost),"from device to host", __FILE__, __LINE__);
    newImage->pixels = newPixels;

    // Free memory
    cudaFree(pixels_dyn);
    cudaFree(newPixels_dyn);

    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for the kernel: %f ms\n", time);

    //Save image
    if(SDL_SavePNG(newImage, "test2.png")) {	//boring way with error checking
	printf("Unable to save png -- %s\n", SDL_GetError());
    }

    // Free surfaces
    SDL_FreeSurface (image);
    SDL_FreeSurface (newImage);
    SDL_Quit();
}
