cmake_minimum_required(VERSION 3.10)

project(cuddh)

configure_file(config.in include/cuddh_config.hpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# LAPACK
find_package(LAPACK REQUIRED)

# specify which version of waveDG to make
option(CUDDH_DEBUG "compiles library in debug mode and add extra checks for correctness (like index bounds, etc.)")

add_compile_options(-Wno-unused-parameter -Wall -Wextra -Wpedantic)
if (CUDDH_DEBUG)
    add_compile_options(-g)
else()
    add_compile_options(-O3) # if not debug then optimize
endif()

add_library(
    cuddh STATIC
    
    cuddh.hpp
    
    source/Basis.cpp
    source/cuddh_error.cpp
    source/Element.cpp
    source/H1Space.cpp
    source/Mesh2D.cpp
    source/QuadratureRule.cpp
)

target_include_directories(cuddh PUBLIC . include)

target_link_libraries(cuddh PUBLIC ${LAPACK_LIBRARIES})

set_target_properties(cuddh PROPERTIES PUBLIC_HEADER cuddh.hpp)

add_subdirectory(tests)