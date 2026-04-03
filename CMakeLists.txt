cmake_minimum_required(VERSION 3.25)

# Project setup
project(cuddh LANGUAGES CUDA CXX)

enable_testing()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA settings
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# Options
option(CUDDH_DEBUG "Enable debug mode" OFF)
option(CUDDH_LOG_MEMCPY "Enable logging of memory copies" OFF)

# Set CUDDH_DEBUG if build type is Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDDH_DEBUG ON)
endif()

# Display option values
message(STATUS "CUDDH_DEBUG: ${CUDDH_DEBUG}")
message(STATUS "CUDDH_LOG_MEMCPY: ${CUDDH_LOG_MEMCPY}")

# Output configuration file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuddh_config.hpp
    @ONLY
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Library setup
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
add_library(cuddh STATIC ${SOURCES})
target_compile_options(cuddh PUBLIC --extended-lambda --expt-relaxed-constexpr -Wall -Wextra -Wno-unused-parameter -Wunused-variable)
set_target_properties(cuddh PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(cuddh PUBLIC cxx_std_17)

# Find and link LAPACK/LAPACKE
find_package(LAPACK REQUIRED)

# LAPACKE can be provided either as liblapacke or bundled in OpenBLAS.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PC_LAPACKE QUIET lapacke)
endif()

find_path(
    LAPACKE_INCLUDE_DIR
    NAMES lapacke.h
    HINTS ${PC_LAPACKE_INCLUDE_DIRS}
    PATH_SUFFIXES openblas
)

find_library(
    LAPACKE_LIBRARY
    NAMES lapacke openblas
    HINTS ${PC_LAPACKE_LIBRARY_DIRS}
)

if(LAPACK_FOUND AND LAPACKE_INCLUDE_DIR AND LAPACKE_LIBRARY)
    target_include_directories(cuddh PUBLIC ${LAPACKE_INCLUDE_DIR})
    target_link_libraries(cuddh PUBLIC ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARY})
else()
    message(FATAL_ERROR "LAPACK/LAPACKE not found. Install LAPACKE development headers and libraries.")
endif()

# Find and link CUDA runtime library (cudart)
find_package(CUDAToolkit 12 REQUIRED)
if(CUDAToolkit_FOUND)
    target_link_libraries(cuddh PUBLIC CUDA::cudart CUDA::cublas)
else()
    message(FATAL_ERROR "CUDAToolkit not found")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(cuddh PRIVATE -O3 --use_fast_math -ffast-math -prec-div=false -prec-sqrt=false)
endif()

# Specify the public header
set_target_properties(cuddh PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/cuddh.hpp)

# Subdirectories
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(benchmarks)
