cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)
project(lpsolver CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS
    "-lineinfo -use_fast_math -Xptxas=-v -Xcompiler=\"${CMAKE_CXX_FLAGS}\"")
set(CMAKE_CUDA_FLAGS_DEBUG "-Xcompiler=\"${CMAKE_CXX_FLAGS_DEBUG}\"")
set(CMAKE_CUDA_FLAGS_RELEASE "-Xcompiler=\"${CMAKE_CXX_FLAGS_RELEASE}\"")

add_library(lpsolver STATIC src/solver.cu)
set_property(TARGET lpsolver PROPERTY CUDA_ARCHITECTURES 70-real 70-virtual)
target_include_directories(lpsolver SYSTEM
                           PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(lpsolver PUBLIC include/)
set_target_properties(lpsolver PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                          CUDA_RESOLVE_DEVICE_SYMBOLS ON)
