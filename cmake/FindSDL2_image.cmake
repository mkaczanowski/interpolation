
# SDL2 Library required
find_package(SDL2 QUIET)
if(NOT SDL2_FOUND)
  set(SDL2_IMAGE_SDL2_NOT_FOUND "Could NOT find SDL2 (SDL2 is required by SDL2_image).")
  if(SDL2_image_FIND_REQUIRED)
    message(FATAL_ERROR ${SDL2_IMAGE_SDL2_NOT_FOUND})
  else()
      if(NOT SDL2_image_FIND_QUIETLY)
        message(STATUS ${SDL2_IMAGE_SDL2_NOT_FOUND})
      endif()
    return()
  endif()
  unset(SDL2_IMAGE_SDL2_NOT_FOUND)
endif()


# Define options for searching SDL2_image Library in a custom path

set(SDL2_IMAGE_PATH "" CACHE STRING "Custom SDL2_image Library path")

set(_SDL2_IMAGE_NO_DEFAULT_PATH OFF)
if(SDL2_IMAGE_PATH)
  set(_SDL2_IMAGE_NO_DEFAULT_PATH ON)
endif()

set(SDL2_IMAGE_NO_DEFAULT_PATH ${_SDL2_IMAGE_NO_DEFAULT_PATH}
    CACHE BOOL "Disable search SDL2_image Library in default path")
unset(_SDL2_IMAGE_NO_DEFAULT_PATH)

set(SDL2_IMAGE_NO_DEFAULT_PATH_CMD)
if(SDL2_IMAGE_NO_DEFAULT_PATH)
  set(SDL2_IMAGE_NO_DEFAULT_PATH_CMD NO_DEFAULT_PATH)
endif()

# Search for the SDL2_image include directory
find_path(SDL2_IMAGE_INCLUDE_DIR SDL_image.h
  HINTS
    ENV SDL2IMAGEDIR
    ENV SDL2DIR
    ${SDL2_IMAGE_NO_DEFAULT_PATH_CMD}
  PATH_SUFFIXES SDL2
                # path suffixes to search inside ENV{SDL2DIR}
                # and ENV{SDL2IMAGEDIR}
                include/SDL2 include
  PATHS ${SDL2_IMAGE_PATH}
  DOC "Where the SDL2_image headers can be found"
)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(VC_LIB_PATH_SUFFIX lib/aarch64)
else()
  set(VC_LIB_PATH_SUFFIX lib/aarch64)
endif()

# Search for the SDL2_image library
find_library(SDL2_IMAGE_LIBRARY
  NAMES SDL2_image
  HINTS
    ENV SDL2IMAGEDIR
    ENV SDL2DIR
    ${SDL2_IMAGE_NO_DEFAULT_PATH_CMD}
  PATH_SUFFIXES lib ${VC_LIB_PATH_SUFFIX}
  PATHS ${SDL2_IMAGE_PATH}
  DOC "Where the SDL2_image Library can be found"
)

# Read SDL2_image version
if(SDL2_IMAGE_INCLUDE_DIR AND EXISTS "${SDL2_IMAGE_INCLUDE_DIR}/SDL_image.h")
  file(STRINGS "${SDL2_IMAGE_INCLUDE_DIR}/SDL_image.h" SDL2_IMAGE_VERSION_MAJOR_LINE REGEX "^#define[ \t]+SDL_IMAGE_MAJOR_VERSION[ \t]+[0-9]+$")
  file(STRINGS "${SDL2_IMAGE_INCLUDE_DIR}/SDL_image.h" SDL2_IMAGE_VERSION_MINOR_LINE REGEX "^#define[ \t]+SDL_IMAGE_MINOR_VERSION[ \t]+[0-9]+$")
  file(STRINGS "${SDL2_IMAGE_INCLUDE_DIR}/SDL_image.h" SDL2_IMAGE_VERSION_PATCH_LINE REGEX "^#define[ \t]+SDL_IMAGE_PATCHLEVEL[ \t]+[0-9]+$")
  string(REGEX REPLACE "^#define[ \t]+SDL_IMAGE_MAJOR_VERSION[ \t]+([0-9]+)$" "\\1" SDL2_IMAGE_VERSION_MAJOR "${SDL2_IMAGE_VERSION_MAJOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_IMAGE_MINOR_VERSION[ \t]+([0-9]+)$" "\\1" SDL2_IMAGE_VERSION_MINOR "${SDL2_IMAGE_VERSION_MINOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_IMAGE_PATCHLEVEL[ \t]+([0-9]+)$" "\\1" SDL2_IMAGE_VERSION_PATCH "${SDL2_IMAGE_VERSION_PATCH_LINE}")
  set(SDL2_IMAGE_VERSION_STRING ${SDL2_IMAGE_VERSION_MAJOR}.${SDL2_IMAGE_VERSION_MINOR}.${SDL2_IMAGE_VERSION_PATCH})
  unset(SDL2_IMAGE_VERSION_MAJOR_LINE)
  unset(SDL2_IMAGE_VERSION_MINOR_LINE)
  unset(SDL2_IMAGE_VERSION_PATCH_LINE)
  unset(SDL2_IMAGE_VERSION_MAJOR)
  unset(SDL2_IMAGE_VERSION_MINOR)
  unset(SDL2_IMAGE_VERSION_PATCH)
endif()

set(SDL2_IMAGE_LIBRARIES ${SDL2_IMAGE_LIBRARY})
set(SDL2_IMAGE_INCLUDE_DIRS ${SDL2_IMAGE_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(SDL2_image
                                  REQUIRED_VARS SDL2_IMAGE_LIBRARIES SDL2_IMAGE_INCLUDE_DIRS
                                  VERSION_VAR SDL2_IMAGE_VERSION_STRING)


mark_as_advanced(SDL2_IMAGE_PATH
                 SDL2_IMAGE_NO_DEFAULT_PATH
                 SDL2_IMAGE_LIBRARY
                 SDL2_IMAGE_INCLUDE_DIR)


if(SDL2_IMAGE_FOUND)

  # SDL2::Image target
  if(SDL2_IMAGE_LIBRARY AND NOT TARGET SDL2::Image)
    add_library(SDL2::Image UNKNOWN IMPORTED)
    set_target_properties(SDL2::Image PROPERTIES
                          IMPORTED_LOCATION "${SDL2_IMAGE_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${SDL2_IMAGE_INCLUDE_DIR}"
                          INTERFACE_LINK_LIBRARIES SDL2::Core)
  endif()
endif()
