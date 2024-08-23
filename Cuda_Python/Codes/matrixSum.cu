#include <cassert>
#include <iostream>
#include <random>
#include <cstdint>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring> // For memcpy

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

int* allocate_on_heap(size_t size){
    return (int*)malloc(size);
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

void initialize(){
    CUresult result = cuInit(0);
}

/*
This is the function that is being
called from python and from this file (cuda).
*/
void calcSum(int* a, int* b, int* result, int len){
    for(int i = 0; i < len; i++){
        result[i] = a[i] + b[i];
    }
}

void calcSumPar(int* a, int* b, int* result, int len);  
void helperPar(int* a, int* b, int* result, int len){
    int* x = (int*)malloc(len * sizeof(int));
    int* y = (int*)malloc(len * sizeof(int));
    int* z = (int*)malloc(len * sizeof(int));

    /*
    for(int i = 0; i < len; i++){
        //std::cout << &a[i] << std::endl;
        x[i] = a[i];
        y[i] = b[i];
        z[i] = result[i];
    }
    */

    std::memcpy(x, a, len * sizeof(int));
    std::memcpy(y, b, len * sizeof(int));
    std::memcpy(z, result, len * sizeof(int));

    calcSumPar(x, y, z, len);
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
    cudaMemcpyAsync(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, b, bytes, cudaMemcpyHostToDevice);

    //Threads per thread block 
    int threadNum_perThreadBl = 256;

    //Number of thread blocks
    int numThreadBlock = (N + threadNum_perThreadBl - 1) / threadNum_perThreadBl;

    //std::cout << "Calling kernel" << std::endl;

    //Launch the kernel
    vectorAdd<<<numThreadBlock, threadNum_perThreadBl>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    //std::cout << "Kernel ended" << std::endl;

    //Copy sum of vectors (c) to host
    cudaMemcpyAsync(result , d_c, bytes, cudaMemcpyDeviceToHost);

    //Verify the sum
    //verify_result(a, b, result, N);  

    //Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main(){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int array_size = 1 << 20; // 2^25 elements
    const int range_max = 2047; // Values in the range 0 to 2047

    // Dynamically allocate the array
    int* result = new int[array_size];

    // Random number generation setup
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> distr(0, range_max); // Define the range

    int loop_size = 10;
    float total_duration = 0;

    for(int i = 0; i < loop_size; i++){

        // Random number generation setup
        std::random_device rd; // Obtain a random number from hardware
        std::mt19937 gen(rd()); // Seed the generator
        std::uniform_int_distribution<> distr(0, range_max); // Define the range

        int* array1 = new int[array_size];
        int* array2 = new int[array_size];

        for (int j = 0; j < array_size; j++) {
            array1[j] = distr(gen);
            array2[j] = distr(gen);
        }

        // Start time
        //auto start = std::chrono::high_resolution_clock::now();
        cudaEventRecord(start);

        //std::cout << "Calling calcSumPar" << std::endl;

        //Operation
        calcSumPar(array1, array2, result, array_size);
        //helperPar(array1, array2, result, array_size);

        //std::cout << "Ended calcSumPar" << std::endl;

        // End time
        //auto end = std::chrono::high_resolution_clock::now();
        cudaEventRecord(stop);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop);

        // Calculate the elapsed time for this iteration
        //auto duration = std::chroncudaDeviceReset();o::duration_cast<std::chrono::microseconds>(end - start).count();

        // Calculate the elapsed time in milliseconds
        float duration;
        cudaEventElapsedTime(&duration, start, stop);
        total_duration += duration;
    
        delete[] array1;
        delete[] array2; 
    }

    std::cout << "Time it took in seconds (miliseconds = 10^3 seconds) (total): " << (total_duration/1000) << std::endl;
    std::cout << "Time it took in seconds (miliseconds = 10^3 seconds) (amortized): " << (total_duration/1000)/loop_size << std::endl;

    return 0;
}