#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>

#include "../Headers/matrixSumCu.h"

//CUDA Kernel
__global__ void vectorAdd(const int *__restrict a,
						  const int *__restrict b,
						  int *__restrict c, int N){
	//Assing the thread id
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	//If within boundaries, sum up
	if(tid < N){
		c[tid] = a[tid] + b[tid];
	}

}

//Check vector add result
//Currently not used to only see the speed of the kernel function
void verify_result(int *a,
				   int *b,
				   int *c, int len){
	for(int i = 0; i < len; i++){
		assert(c[i] == a[i] + b[i]);
	}

}

void calcSumPar(int* a, int* b, int* result, int len){
    //Arrays to be summed are size of len
	int N = len; //Thread amount
	size_t bytes = sizeof(int) * N; 

	//Vectors for holding the host-side (CPU) data

    //Allocate memory on device (GPU)
    int *d_a, *d_b, *d_c;
  
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    //Copy data from host to device (CPU -> GPU)
    //Memory location of d_a (pointer) copied bytes amount of data from a (pointer)
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    //Threads per thread block 
    int threadNum_perThreadBl = 256;

    //Number of thread blocks
    int numThreadBlock = (N + threadNum_perThreadBl - 1) / threadNum_perThreadBl;

    //Launch the kernel
    vectorAdd<<<numThreadBlock, threadNum_perThreadBl>>>(d_a, d_b, d_c, N);

    //Copy sum of vectors (c) to host
    cudaMemcpy(result , d_c, bytes, cudaMemcpyDeviceToHost);
  
    //Verify the sum
 	//verify_result(a, b, c, N);  

    //Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //std::cout << "Assignment done in parallel" << std::endl;
}