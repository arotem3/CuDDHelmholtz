cmake_minimum_required(VERSION 3.18)

file(GLOB_RECURSE TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(test ${TEST_SOURCES})
set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)

target_compile_definitions(test PRIVATE UNSTRUCTURED_SQUARE_MESH_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../meshes/unstructured_square")

target_link_libraries(test PRIVATE cuddh)