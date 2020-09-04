cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(maximum_flow CXX CUDA)

add_executable(maximum_flow maximum_flow.cpp)
target_link_libraries(maximum_flow PRIVATE lpsolver)
