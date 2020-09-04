cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(minimum_distace CXX CUDA)

add_executable(minimum_distance minimum_distance.cpp)
target_link_libraries(minimum_distance PRIVATE lpsolver)
