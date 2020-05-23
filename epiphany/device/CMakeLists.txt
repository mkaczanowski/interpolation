project(interpolation)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_SYSTEM_NAME epiphany)
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

set(CMAKE_C_COMPILER   /opt/adapteva/esdk/tools/e-gnu/bin/e-gcc)
set(CMAKE_CXX_COMPILER /opt/adapteva/esdk/tools/e-gnu/bin/e-g++)

set(CMAKE_CXX_STANDARD 14)
set_source_files_properties( ../../lib/lib.c PROPERTIES LANGUAGE C )

# add epiphany libs
set(E_SDK /opt/adapteva/esdk)
include_directories( ${ESDK}/tools/e-gnu.armv7l/epiphany-elf/include )
find_library(E_LIB e-lib HINTS ${ESDK}/tools/e-gnu.armv7l/epiphany-elf/lib)

add_compile_options(-Wall -Wextra -g -O2)

add_link_options(-Wall -Wextra -g  -T /opt/adapteva/esdk/bsps/current/fast.ldf)

add_executable(
    e_interpolation
    e_interpolation.c
    ../../lib/lib.c
    ../../lib/lib.h
)

target_link_libraries(
    e_interpolation
    ${E_LIB}
)
