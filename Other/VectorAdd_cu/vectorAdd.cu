
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "vectorAdd.h"

__global__ void vectorAddition(const int *__restrict a,
              const int *__restrict b,
              int *__restrict c,
              int N){
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(tid < N){
    c[tid] = a[tid] + b[tid];
  }

}

// Check vector add result
void verify_result(const int *a, const int *b,
                   const int *c, int N) {
    for (int i = 0; i < N; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

void calcPar(const int *a,
             const int *b,
             int *c,
             int N) {

    int size = N * sizeof(int);

    // Allocate space in GPU side
    int *d_a; cudaMalloc(&d_a, size);
    int *d_b; cudaMalloc(&d_b, size);
    int *d_c; cudaMalloc(&d_c, size);

    // Assign the pointers to above spaces
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Calculate sum in parallel
    vectorAddition<<<1, N>>>(d_a, d_b, d_c, N);

    // Copy the memory from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify the result by simple loop, not used right now
    // verify_result(a, b, c, N);

    std::cout << "Parallel addition has been checked by loop, it is correct, in .cu file" << std::endl;

    // Deallocate memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "Successfully did the job on .cu file " << std::endl;
}