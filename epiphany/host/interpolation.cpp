#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <e-hal.h>
#include <e-loader.h>
#include <unistd.h>

#include "../../png/savepng.h"
#include "../../lib/lib.h"

using namespace cv;
using namespace std;

e_platform_t platform;
e_epiphany_t dev;

e_mem_t	old_mbuf;
e_mem_t	new_mbuf;
const char old_shm_name[] = "old_pixels_shm"; 
const char new_shm_name[] = "new_pixels_shm"; 

const uint32_t amask = 0xff000000;
const uint32_t rmask = 0x00ff0000;
const uint32_t gmask = 0x0000ff00;
const uint32_t bmask = 0x000000ff;

void processImage(SDL_Surface *image, SDL_Surface **newImage, int rWidth) {
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
    if (old_mbuf.memfd == 0) {
		int rc = e_shm_alloc(&old_mbuf, old_shm_name, imageByteLength);
		if (rc != E_OK)
			rc = e_shm_attach(&old_mbuf, old_shm_name);
    }

	int n = e_write(&old_mbuf, 0, 0, 0, (void*) pixels, imageByteLength);
	if (n != imageByteLength || imageByteLength == 0) {
		cout << "failed to copy image to shared memory, imageByteLength: " << imageByteLength << endl;
		return;
	}

    // initialize shared memory region
    if (new_mbuf.memfd == 0) {
		int rc = e_shm_alloc(&new_mbuf, new_shm_name, newImageByteLength);
		if (rc != E_OK)
			rc = e_shm_attach(&new_mbuf, new_shm_name);
    }

    // start measuring time
    auto start = std::chrono::high_resolution_clock::now(); 

	// core state
	int** cores = (int**) calloc(platform.rows, sizeof(int*));
	for(int i = 0; i < 100; ++i) {
		cores[i] = (int*) calloc(platform.cols, sizeof(int));
	}

	for (unsigned row = 0; row < platform.rows; row++) {
		for (unsigned col = 0; col < platform.cols; col++) {
    		// write args to epiphany
			e_write(&dev, row, col, 0x7000, &((*newImage)->pitch), sizeof(uint16_t)); // pitchOutput
			e_write(&dev, row, col, 0x7004, &(image->pitch), sizeof(uint16_t)); // pitchInput
			e_write(&dev, row, col, 0x7008, &imageByteLength, sizeof(int)); // imageByteLength
			e_write(&dev, row, col, 0x7012, &newImageByteLength, sizeof(int)); // newByteLength
			e_write(&dev, row, col, 0x7016, &newWidth, sizeof(int)); // newWidth
			e_write(&dev, row, col, 0x7020, &newHeight, sizeof(int)); // newHeight
			e_write(&dev, row, col, 0x7024, &xRatio, sizeof(float)); // xRatio
			e_write(&dev, row, col, 0x7028, &yRatio, sizeof(float)); // xRatio

			// unlock work on the core
			e_write(&dev, row, col, 0x7032, &(cores[row][col]), sizeof(uint8_t));
		}
	}

	// wait for jobs to finish
	while (1) {
		unsigned skipped = 0;
		for (unsigned row = 0; row < platform.rows; row++) {
			for (unsigned col = 0; col < platform.cols; col++) {
				if (cores[row][col] == 1) {
					skipped++;
				}

				e_read(&dev, row, col, 0x7032, &(cores[row][col]), sizeof(int));
			}
		}

		if (skipped == (platform.rows * platform.cols)) {
			break;
		}
	}

    // stop the timer
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start
    ); 

	// int debug;
	// e_read(&dev, 0, 0, 0x7036, &debug, sizeof(int));
	// cout << "DEBUG " << debug << endl;

	// copy scaled image
	e_mem_t *newPixelsPmem = (e_mem_t*) &new_mbuf;
	uint8_t* newPixels = (uint8_t *) newPixelsPmem->base;
    (*newImage)->pixels = newPixels;

    cout << "Time for the kernel: " << duration.count() << " microseconds" << endl; 
}

int main(void) {
    e_init(NULL);
    e_reset_system();
    e_get_platform_info(&platform);

	e_open(&dev, 0, 0, platform.rows, platform.cols);
    if ( E_OK != e_load_group("../../device/build/e_interpolation", &dev, 0, 0, platform.rows, platform.cols, E_TRUE) ) {
		cout << "Error while loading epiphany program" << endl;
		return -1;
    }

    VideoCapture cap("../../../assets/video.mp4"); 
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat frame;
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
        processImage(image, &newImage, 1100);

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
    }

    cap.release();

    e_shm_release(old_shm_name);
    e_shm_release(new_shm_name);
    e_finalize();
    e_close(&dev);

    SDL_FreeSurface(newImage); // this will throw free() error because of newPixels is located on shared memory
    SDL_Quit();
}
