cmake_minimum_required(VERSION 2.8)
find_package(CUDA)
find_package(PythonLibs 2.7 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

cuda_add_library(gpu_library SHARED
  gpu_library.cpp
  gpu_library.cu)

target_link_libraries(gpu_library
  ${PYTHON_LIBRARIES}
  cudart)

set_target_properties(gpu_library PROPERTIES PREFIX "")
