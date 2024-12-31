cmake_minimum_required(VERSION 3.18)

# Project setup
project(cuddh LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA settings
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --compiler-options -Wall,-Wextra,-Wno-unused-parameter,-Wunused-variable")

# Options
option(CUDDH_DEBUG "Enable debug mode" OFF)
option(CUDDH_LOG_MEMCPY "Enable logging of memory copies" OFF)

# Set compile options based on CUDDH_DEBUG
if(CUDDH_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

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

# Find and link LAPACK
find_package(LAPACK REQUIRED)
if(LAPACK_FOUND)
    message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
    target_link_libraries(cuddh PRIVATE ${LAPACK_LIBRARIES})
else()
    message(FATAL_ERROR "LAPACK not found")
endif()

# Specify the public header
set_target_properties(cuddh PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/cuddh.hpp)

# Subdirectories
add_subdirectory(tests)
add_subdirectory(examples)
