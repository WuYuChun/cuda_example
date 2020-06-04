cmake_minimum_required(VERSION 2.8.0)
project(cuda_example)

find_package(CUDA REQUIRED)


if(HAS_DIAG_COLOR_FLAG)
  if(CMAKE_GENERATOR MATCHES "Ninja")
    add_compile_options(-fdiagnostics- color=always)
  else()
    add_compile_options(-fdiagnostics-color=auto)
  endif()
endif()

set(CUDA_NVCC_FLAGS -G;-g)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory(src/simple_add)
add_subdirectory(src/bitreverse)
#add_subdirectory(src/autostep)
add_subdirectory(src/cuda_memcheck_demo)
add_subdirectory(src/block_error)
add_subdirectory(src/initcheck)
add_subdirectory(src/synccheck)
add_subdirectory(common_tool)
