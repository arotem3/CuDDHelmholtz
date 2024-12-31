cmake_minimum_required(VERSION 3.18)

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Gather all .cpp files in the examples directory
file(GLOB_RECURSE EXAMPLE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Create an executable for each .cpp file
foreach(EXAMPLE_FILE ${EXAMPLE_SOURCES})
    # Get the filename without directory and extension
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WE)

    # Add executable target
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_FILE})

    # Set the source file to be treated as CUDA code
    set_source_files_properties(${EXAMPLE_FILE} PROPERTIES LANGUAGE CUDA)

    # Link against cuddh
    target_link_libraries(${EXAMPLE_NAME} PRIVATE cuddh)

    # Add post-build command to create the ../solution directory
    add_custom_command(TARGET ${EXAMPLE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/../solution
    )
endforeach()
