cmake_minimum_required(VERSION 3.18)

enable_testing()

# List of test files
file(GLOB TEST_FILES_TEMP "*.cpp")
list(TRANSFORM TEST_FILES_TEMP REPLACE ".cpp$" "" OUTPUT_VARIABLE TEST_FILES)
foreach(FILE ${TEST_FILES})
    get_filename_component(TEST_NAME "${FILE}" NAME)
    list(APPEND TEST_NAMES "${TEST_NAME}")
endforeach()
set(TEST_FILES "${TEST_NAMES}")

# Create an executable for each test
foreach(TEST_NAME ${TEST_FILES})
    add_executable(test_${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cpp)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(test_${TEST_NAME} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    
    target_compile_definitions(test_${TEST_NAME} PRIVATE UNSTRUCTURED_SQUARE_MESH_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../meshes/unstructured_square")
    
    target_link_libraries(test_${TEST_NAME} PRIVATE cuddh)
    
    # Register with ctest
    add_test(NAME ${TEST_NAME} COMMAND test_${TEST_NAME})
endforeach()