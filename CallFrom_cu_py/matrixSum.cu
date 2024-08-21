#include <random>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

//CUDA Kernel
__global__ void vectorAdd(int* a,
						  int* b,
						  int* c, int N){
	//Assing the thread id
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	//If within boundaries, sum up
	if(tid < N){
		c[tid] = a[tid] + b[tid];
	}

}

void resetDevice(){
    cudaDeviceReset();
}

/*
Verification for the results.
*/
void verify_result(int *a,
				   int *b,
				   int *c, int len){
	for(int i = 0; i < len; i++){
		assert(c[i] == a[i] + b[i]);
	}
}

void funcX(){
    //Size of the vector addition
    int N = 1 << 20;

    size_t bytes = N * sizeof(int);

    int* a;
    int* b;
    int* c;

    a = (int*)malloc(bytes);
    b = (int*)malloc(bytes);
    c = (int*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

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
    int threadNum_perThreadBl = 1024;

    //Number of thread blocks
    int numThreadBlock = (N + threadNum_perThreadBl - 1) / threadNum_perThreadBl;

    //Launch the kernel
    vectorAdd<<<numThreadBlock, threadNum_perThreadBl>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    //Copy sum of vectors (c) to host
    cudaMemcpy(c , d_c, bytes, cudaMemcpyDeviceToHost);

    //Validate Result
    verify_result(a, b, c, N);

    //Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
}

int main(){
    cudaDeviceReset();

    //(1)
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    */

    float total_duration = 0;
    float duration = 0;

    int repetition = 100;

    for(int i = 0; i < repetition; i++){
        //Start the clock
        //(1)
        //cudaEventRecord(start);
        //(2)
        auto start = std::chrono::high_resolution_clock::now();

        //funcX call
        funcX();

        // End time
        //(1)
        //cudaEventRecord(stop);
        //(2)
        auto end = std::chrono::high_resolution_clock::now();

        //Wait for the stop event to complete
        //(1)
        //cudaEventSynchronize(stop);

        //(2)
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        //Calculate the elapsed time in milliseconds
        //(1)
        //cudaEventElapsedTime(&duration, start, stop);

        total_duration += duration;
    }

    std::cout << "Calculations are correct" << std::endl;
    std::cout << "Time it took in seconds (miliseconds * 10^3 =  seconds): " << (total_duration/1000) << "\n" << std::endl;
    std::cout << "Amortized time in seconds (miliseconds * 10^3 =  seconds): " << (total_duration/1000) / repetition << "\n" << std::endl;
}