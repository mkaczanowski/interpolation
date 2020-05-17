project(e_interpolation C)
cmake_minimum_required(VERSION 2.8)

set_source_files_properties( ../../lib/lib.c PROPERTIES LANGUAGE C )

SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_C_COMPILER   /opt/adapteva/esdk/tools/e-gnu/bin/epiphany-elf-gcc)
SET(CMAKE_CXX_COMPILER /opt/adapteva/esdk/tools/e-gnu/bin/epiphany-elf-g++)
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)
SET(CMAKE_C_LINK_EXECUTABLE "${CMAKE_C_LINK_EXECUTABLE} -le-lib")

include_directories( /opt/adapteva/esdk/tools/e-gnu.armv7l/epiphany-elf/include )
link_directories( /opt/adapteva/esdk/tools/e-gnu.armv7l/epiphany-elf/lib )

add_executable(
    e_interpolation
    e_interpolation.c
    ../../lib/lib.c
    ../../lib/lib.h
)

target_link_libraries(
    e_interpolation
)
