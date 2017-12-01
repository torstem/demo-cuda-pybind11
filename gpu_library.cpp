#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

void run_kernel
(double *vec, double scalar, int num_elements);

void multiply_with_scalar(pybind11::array_t<double> vec, double scalar)
{
  int size = 10;
  double *gpu_ptr;
  cudaError_t error = cudaMalloc(&gpu_ptr, size * sizeof(double));

  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
  auto ha = vec.request();

  if (ha.ndim != 1) {
    std::stringstream strstr;
    strstr << "ha.ndim != 1" << std::endl;
    strstr << "ha.ndim: " << ha.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }

  double* ptr = reinterpret_cast<double*>(ha.ptr);
  error = cudaMemcpy(gpu_ptr, ptr, size * sizeof(double), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  run_kernel(gpu_ptr, scalar, size);

  error = cudaMemcpy(ptr, gpu_ptr, size * sizeof(double), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  error = cudaFree(gpu_ptr);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

PYBIND11_MODULE(gpu_library, m)
{
  m.def("multiply_with_scalar", multiply_with_scalar);
}
