cmake_minimum_required(VERSION 3.25)

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

file(GLOB_RECURSE BENCHMARK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Create an executable for each .cpp file
foreach(BENCHMARK_FILE ${BENCHMARK_SOURCES})
    get_filename_component(BENCHMARK_NAME ${BENCHMARK_FILE} NAME_WE)
    add_executable(${BENCHMARK_NAME} ${BENCHMARK_FILE})
    set_source_files_properties(${BENCHMARK_FILE} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${BENCHMARK_NAME} PRIVATE cuddh)
endforeach()
