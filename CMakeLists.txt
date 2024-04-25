cmake_minimum_required(VERSION 3.10)

project(cuddh LANGUAGES CXX CUDA)

configure_file(config.in include/cuddh_config.hpp)

set(CMAKE_CXX_COMPILER nvcc)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 --extended-lambda -arch=sm_80 --compiler-options -Wall,-Wextra,-Wno-unused-parameter")

# LAPACK -- something funny is happening here with find package
# find_package(LAPACK REQUIRED)
# find_package(BLAS REQUIRED)

# specify which version of waveDG to make
option(CUDDH_DEBUG "compiles library in debug mode and add extra checks for correctness (like index bounds, etc.)")
option(CUDDH_LOG_MEMCPY "HostDeviceArray will log every cuda memcpy to stdout")

if (CUDDH_DEBUG)
    add_compile_options(-g)
else()
    add_compile_options(-O3)
endif()

add_library(
    cuddh STATIC
    
    cuddh.hpp
    
    source/Basis.cpp
    source/cuddh_error.cpp
    # source/DDH.cpp
    source/Element.cpp
    source/EnsembleSpace.cpp
    source/FaceLinearFunctional.cpp
    source/FaceMassMatrix.cpp
    source/gmres.cpp
    source/H1Space.cpp
    source/linalg.cpp
    source/LinearFunctional.cpp
    source/MassMatrix.cpp
    source/Mesh2D.cpp
    source/QuadratureRule.cpp
    source/StiffnessMatrix.cpp
)

target_compile_options(cuddh PRIVATE -x=cu -dc)
target_include_directories(cuddh PUBLIC . include)

target_link_options(cuddh PUBLIC -llapack -lblas) # temp fix
# target_link_libraries(cuddh PUBLIC ${LAPACK_LIBRARIES})

set_target_properties(cuddh PROPERTIES PUBLIC_HEADER cuddh.hpp)

add_subdirectory(tests)
add_subdirectory(examples)