cmake_minimum_required(VERSION 3.25)
project(preAApp)

set(CMAKE_CXX_STANDARD 14)

# Set include directories
set(INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/examples
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

# Find and set Eigen3
find_package(Eigen3 REQUIRED)

# Print Eigen3 version information
message("-- Eigen3 version: ${EIGEN3_VERSION_STRING}")

# Retrieve all files in the examples folder
file(GLOB EXAMPLE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp)

# Create an empty list of example names
set(EXAMPLE_NAMES "")

# Iterate over each example file and add its name to the list
foreach(EXAMPLE_FILE IN LISTS EXAMPLE_FILES)
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WE)
    list(APPEND EXAMPLE_NAMES ${EXAMPLE_NAME})
endforeach()

# Define a CMake function to add an example
function(add_example EXAMPLE_NAME)
    add_executable(${EXAMPLE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/examples/${EXAMPLE_NAME}.cpp)
    target_include_directories(${EXAMPLE_NAME} PRIVATE ${INCLUDE_DIRS})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE Eigen3::Eigen)
endfunction()

# Iterate over example names and call the add_example function
foreach(EXAMPLE_NAME IN LISTS EXAMPLE_NAMES)
    add_example(${EXAMPLE_NAME})
endforeach()