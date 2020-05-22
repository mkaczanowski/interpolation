project(interpolation)
cmake_minimum_required(VERSION 2.8)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(PNG REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIRS} ${PNG_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 14)
set_source_files_properties( ../../lib/lib.c PROPERTIES LANGUAGE CXX )

# add epiphany libs
include_directories( /opt/adapteva/esdk/tools/host/include )
find_library(E_HAL e-hal HINTS /opt/adapteva/esdk/tools/host/lib)
find_library(E_LOADER e-loader HINTS /opt/adapteva/esdk/tools/host/lib)

add_compile_options(-Wall -Wextra)

add_executable(
    interpolation
    interpolation.cpp
    ../../lib/lib.c
    ../../lib/lib.h
    ../../png/savepng.c
    ../../png/savepng.h
)

target_link_libraries(
    interpolation
    ${SDL2_LIBRARIES}
    ${SDL2_IMAGE_LIBRARIES}
    ${PNG_LIBRARY}
    ${OpenCV_LIBS}
	${E_HAL}
	${E_LOADER}
)
