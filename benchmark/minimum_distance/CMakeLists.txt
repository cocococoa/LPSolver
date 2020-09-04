cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(minimum_distace CXX CUDA)

set(OR_TOOLS_PATH
    ${CMAKE_SOURCE_DIR}/externals/or-tools_Ubuntu-18.04-64bit_v7.8.7959)

add_executable(minimum_distance minimum_distance.cpp)
target_include_directories(minimum_distance PRIVATE ${OR_TOOLS_PATH}/include)
target_link_libraries(
  minimum_distance PRIVATE ${OR_TOOLS_PATH}/lib/libortools.so
                           ${OR_TOOLS_PATH}/lib/libglog.so lpsolver)
