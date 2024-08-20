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
    //std::cout<< N << std::endl;

    //std::cout << "In funcX" << std::endl;

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

    //std::cout << "Call gpu" << std::endl;
    //Launch the kernel
    vectorAdd<<<numThreadBlock, threadNum_perThreadBl>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    //std::cout << "Ended gpu" << std::endl;

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

    //std::cout << "Out func" << std::endl;
}

int main(){
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_duration = 0;
    float duration = 0;

    int repetition = 100;
    */

    //float record[repetition];

    //for(int i = 0; i < 100; i++){
        //Start the clock
        //cudaEventRecord(start);

        //std::cout << "Start the clock" << std::endl;
        //auto start = std::chrono::high_resolution_clock::now();
        //funcX call
    funcX();

        // End time
        //cudaEventRecord(stop);
        //auto end = std::chrono::high_resolution_clock::now();

        // Wait for the stop event to complete
        //cudaEventSynchronize(stop);
        //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //std::cout << "Stop the clock" << std::endl;

        // Calculate the elapsed time in milliseconds
        //cudaEventElapsedTime(&duration, start, stop);

        //total_duration += duration;
        //record[i] = duration;
    //}

    //std::cout << "Calculations are correct" << std::endl;
    //std::cout << "Time it took in miliseconds (miliseconds * 10^3 =  seconds): " << (total_duration) << "\n" << std::endl;

    /*
    for(int i = 0; i < repetition; i++){
        std::cout << record[i] << std::endl;
    }
    */
}