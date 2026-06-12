cmake_minimum_required(VERSION 3.18)

project(cuda_learn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

function(add_cuda_example target source)
  add_executable(${target} ${source})
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endfunction()

add_subdirectory(src/cuda_basics)
