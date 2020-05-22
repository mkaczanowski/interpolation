#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>

#include "../png/savepng.h"
#include "../lib/lib.h"

using namespace cv;
using namespace std;

const uint32_t amask = 0xff000000;
const uint32_t rmask = 0x00ff0000;
const uint32_t gmask = 0x0000ff00;
const uint32_t bmask = 0x000000ff;

void processImage(SDL_Surface *image, SDL_Surface **newImage, uint8_t **oldPixels, uint8_t **newPixels, int rWidth) {
    int imageByteLength = image->w * image->h * sizeof(uint8_t)*image->format->BytesPerPixel;

    int newWidth = image->w + (rWidth - image->w);
    int newHeight = image->h + (rWidth - image->w);

    // create scaled image surface
    if (*newImage == NULL) {
        *newImage = SDL_CreateRGBSurface(SDL_SWSURFACE, newWidth, newHeight, 32, rmask, gmask, bmask, amask);
    }
    int newImageByteLength = newWidth * newHeight * sizeof(uint8_t)* (*newImage)->format->BytesPerPixel;

    float xRatio = ((float)(image->w-1)) / (*newImage)->w;
    float yRatio = ((float)(image->h-1)) / (*newImage)->h;

    // create pointer to device and host pixels
    uint8_t *pixels = (uint8_t*) image->pixels;

    // copy original image to device memory
    if (*oldPixels == NULL) {
        *oldPixels = (uint8_t*) pixels;
    }

    // initialize shared memory region
    if (*newPixels == NULL) {
        *newPixels = (uint8_t*) malloc(newImageByteLength);
    }

    // start measuring time
    auto start = std::chrono::high_resolution_clock::now(); 

    // bilinear transform on CUDA device
    for (int x = 0; x < newWidth; x++) {
        for (int y = 0; y < newHeight; y++) {
            bilinearTransform(x, y, *newPixels, *oldPixels, (*newImage)->pitch, image->pitch, image->format->BytesPerPixel, (*newImage)->format->BytesPerPixel, xRatio, yRatio);
        }
    }

    // stop the timer
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start
    ); 

    // copy scaled image to host
    (*newImage)->pixels = *newPixels;

    cout << "Time for the kernel: " << duration.count() << " microseconds" << endl; 
}

int main(void) {
    VideoCapture cap("../../assets/video.mp4"); 

    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat frame;
    uint8_t *oldPixels = NULL; // device memory
    uint8_t *newPixels = NULL; // shared memory
    SDL_Surface *newImage = NULL;

    int i = 0;
    while(1){
        // start timing
        auto start = std::chrono::high_resolution_clock::now(); 

        // read frame from a video stream
        cap >> frame;

        // select 5 frames in the middle
        if (i++ < 100) {
            continue;
        } else if( i > 105)  {
            break;
        }

        if (frame.empty()) {
            break;
        }

        // convert opencv frame to SDL surface
        IplImage opencvimg = cvIplImage(frame);
        SDL_Surface *image = SDL_CreateRGBSurfaceFrom((void*)opencvimg.imageData,
                opencvimg.width,
                opencvimg.height,
                opencvimg.depth * opencvimg.nChannels,
                opencvimg.widthStep,
                rmask, gmask, bmask, amask
        );

        if (!image){
            printf("IMG_Load: %s\n", IMG_GetError());
            return 255;
        }

        // run bilinear transformation
        processImage(image, &newImage, &oldPixels, &newPixels, 1500);

        // end timing
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start
        ); 

        cout << "Time taken by function: " << duration.count() << " microseconds" << endl;

        // save new image to PNG
        if(SDL_SavePNG(newImage, "output.png")) {
            cout << "Unable to save png: " << SDL_GetError() << endl;
        }

        SDL_FreeSurface(image);
		break;
    }

    cap.release();
    free(oldPixels);
    free(newPixels);

    SDL_FreeSurface(newImage); // this will throw free() error because of newPixels is located on shared memory
    SDL_Quit();
}
