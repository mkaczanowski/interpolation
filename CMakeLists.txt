cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(PNG REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIRS} ${PNG_INCLUDE_DIR})

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 
)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
	interpolation
	interpolation.cu
	png/savepng.c
	png/savepng.h
)

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    interpolation
    ${SDL2_LIBRARIES}
    ${SDL2_IMAGE_LIBRARIES}
    ${PNG_LIBRARY}
)
