cmake_minimum_required(VERSION 3.25)
project(preAApp)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(examples)
include_directories(include)

# + ----------------- +
# | BUILDING SETTINGS |
# + ----------------- +

#if(NOT CMAKE_BUILD_TYPE)
#    set(CMAKE_BUILD_TYPE Release)
#endif(NOT CMAKE_BUILD_TYPE)
#
## + ----------------- +
## | COMPILATION FLAGS |
## + ----------------- +
#
#include(CheckCXXCompilerFlag)
#check_cxx_compiler_flag(-Wall WALL_SUPPORTED)
#if(${WALL_SUPPORTED})
#    add_compile_options(-Wall)
#endif()
#
## + --------------- +
## | preAApp LIBRARY |
## + --------------- +
#
#add_library(preAApp INTERFACE)
#
## + -------- +
## | INCLUDES |
## + -------- +
#
#target_include_directories(preAApp INTERFACE
#        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
#        )
#
## + ----------------------- +
## | FIND EXTERNAL LIBRARIES |
## + ----------------------- +
#
#find_package(Eigen3 3.4.0 REQUIRED)
#target_link_libraries(preAApp INTERFACE Eigen3::Eigen)
#message("-- Eigen3 version: " ${EIGEN3_VERSION_STRING})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
message("-- Eigen3 version: " ${EIGEN3_VERSION_STRING})
#
## + ------------ +
## | INSTALLATION |
## + ------------ +
#
## Copy headers folder
#install(
#        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
#        DESTINATION include
#)
## Create an export set
#install(TARGETS preAApp EXPORT preAAppTargets)
#
#include(CMakePackageConfigHelpers)
## Version file
#write_basic_package_version_file(
#        ${CMAKE_CURRENT_BINARY_DIR}/preAApp-config-version.cmake
#        VERSION ${PROJECT_VERSION}
#        COMPATIBILITY AnyNewerVersion
#)
## Config file
#configure_package_config_file(
#        ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in
#        ${CMAKE_CURRENT_BINARY_DIR}/preAApp-config.cmake
#        INSTALL_DESTINATION lib/cmake/preAApp
#)
## Targets files
#export(
#        EXPORT preAAppTargets
#        FILE ${CMAKE_CURRENT_BINARY_DIR}/preAApp-targets.cmake
#)
#install(
#        EXPORT preAAppTargets
#        FILE preAApp-targets.cmake
#        DESTINATION lib/cmake/preAApp
#)
#install(
#        FILES
#        ${CMAKE_CURRENT_BINARY_DIR}/preAApp-config.cmake
#        ${CMAKE_CURRENT_BINARY_DIR}/preAApp-config-version.cmake
#        DESTINATION lib/cmake/preAApp
#)

add_executable(rosenbrock_example examples/rosenbrock_example.cpp)
