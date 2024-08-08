#include <cassert>
#include <iostream>
#include <random>
#include <cstdint>
#include <chrono>

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

/*
This is the function that is being
called from python and from this file (cuda).
*/
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

int main(){

    const int array_size = 1 << 25; // 2^25 elements
    const int range_max = 2047; // Values in the range 0 to 2047

    // Dynamically allocate the array
    int* result = new int[array_size];

    // Random number generation setup
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> distr(0, range_max); // Define the range

    int loop_size = 100;
    long long total_duration = 0;

    //long long overhead_duration = 0;
    /* //Overhead calculation
    for (int i = 0; i < loop_size; ++i) {
        auto overhead_start = std::chrono::high_resolution_clock::now();
        auto overhead_end = std::chrono::high_resolution_clock::now();
        overhead_duration += std::chrono::duration_cast<std::chrono::microseconds>(overhead_end - overhead_start).count();
    }  */

    for(int i = 0; i < loop_size; i++){
        int* array1 = new int[array_size];
        int* array2 = new int[array_size];

        for (int j = 0; j < array_size; j++) {
            array1[j] = distr(gen);
        }
        for (int j = 0; j < array_size; j++) {
            array2[j] = distr(gen);
        }

        // Start time
        auto start = std::chrono::high_resolution_clock::now();

        //Operation
        calcSumPar(array1, array2, result, array_size);

        // End time
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate the elapsed time for this iteration
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        total_duration += duration;
    
        delete[] array1;
        delete[] array2; 
    }

    std::cout << "Time it took in seconds (seconds = 10^6 microseconds): " << (total_duration)*pow(10, 6) << std::endl;

    return 0;
}