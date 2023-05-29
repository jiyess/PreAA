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

# Add executable
add_executable(rosenbrock_example ${CMAKE_CURRENT_SOURCE_DIR}/examples/rosenbrock_example.cpp)
add_executable(simple_example ${CMAKE_CURRENT_SOURCE_DIR}/examples/simple_example.cpp)

# Set include directories
target_include_directories(rosenbrock_example PRIVATE ${INCLUDE_DIRS})
target_include_directories(simple_example PRIVATE ${INCLUDE_DIRS})

# Link Eigen3 library
target_link_libraries(rosenbrock_example PRIVATE Eigen3::Eigen)
target_link_libraries(simple_example PRIVATE Eigen3::Eigen)
