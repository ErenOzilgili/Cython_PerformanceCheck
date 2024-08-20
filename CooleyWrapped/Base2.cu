#include <iostream>
#include <stdio.h>
#include <cooperative_groups.h>
#include <complex>
#include <cmath>
#include "Base2.cuh"

using std::cout, std::endl;
using namespace cooperative_groups;
namespace cg = cooperative_groups;

__host__ int bitReverse(int toReverse, int n){
	int exponent = std::log2(n) - 1;
	int reversedInt = 0;
	for(int bitPos = 0; bitPos < std::log2(n); bitPos++){
		bool bitBool = (toReverse >> bitPos) & 1; //Extract the bit in position bitPos
		int bitInt = (int)bitBool;

		reversedInt += bitInt * pow(2, exponent);
		exponent -= 1;
	}

	return reversedInt;
}

__device__ inline float2 addFloat2(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ inline float2 subFloat2(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ inline float2 mulFloat2(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); // Complex multiplication
}

__device__ inline float2 normFloat2(float2 a, float normalize){
    return make_float2(a.x / normalize, a.y / normalize);
}

__device__ void cooleyButterOp(float2 &upp, float2 &low, float2 root){
    float2 rootAndVal = mulFloat2(low, root);//root times one of the values

    float2 inUpp = addFloat2(upp, rootAndVal);
    float2 inLow = subFloat2(upp, rootAndVal);

    upp = inUpp;
    low = inLow;

    //printf("Current thread id: % d -- real part of new upp: %f  imaginary part of new upp %f\n", currTid, upp.x, upp.y);
    //printf("Current thread id: % d -- real part of new low: %f  imaginary part of new low %f\n", currTid, low.x, low.y);
}

__device__ inline void invCooleyNormalize(int tid, int offset, float2* input){
    //When we normalize, offset will be equal to n/2, we want to normalize with n.
    input[tid] = normFloat2(input[tid], 2*offset);
    input[tid + offset] = normFloat2(input[tid+offset], 2*offset);
}

__global__ void parButterFlies(int num, float2* input, bool coeffToVec){
    //Will have total of num/2 threads since we are doing radix-2 Cooley Tukey.
    int logNum = __float2int_rz(log2f((float)num));  // Using log2f for single-precision log2
    int offset = 1;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int currTid = tid;

    thread_block tb = this_thread_block();

    extern __shared__ float2 sharedMem[];

    //Load the part of the global memory for each thread.
    sharedMem[tid] = input[tid]; sharedMem[(num/2) + tid] = input[(num/2) + tid];

    //Synchronise to ensure the data is put into sharedMemory entirely.
    tb.sync();
    
    //Start performing the stages, if coeffToVec=true FFT and if not inverse FFT.
    if(coeffToVec){
        for(int i = 0; i < logNum; i++){
            currTid = tid + ((tid / offset) * offset);

            // Calculate the exponent and twiddle factor
            float angle = (tid % offset) * (-2.0f) * M_PI / (2 * offset);
            float2 rootCom = make_float2(cosf(angle), sinf(angle)); 

            //Perform the butterfly operations
            cooleyButterOp(sharedMem[currTid], sharedMem[currTid + offset], rootCom);

            offset *= 2;
            tb.sync();
        }
    }
    else{
        for(int i = 0; i < logNum; i++){
            currTid = tid + ((tid / offset) * offset);

            // Calculate the exponent and twiddle factor
            float angle = (tid % offset) * (2.0f) * M_PI / (2 * offset);
            float2 rootCom = make_float2(cosf(angle), sinf(angle)); 

            // Perform the butterfly operations
            cooleyButterOp(sharedMem[currTid], sharedMem[currTid + offset], rootCom);

            offset *= 2;
            tb.sync();
        }
        //Divide by factor of n. Notice that offset is n by the end of the loop.
        invCooleyNormalize(tid, offset/2, sharedMem); //So we need to pass the half of the current offset.
    }

    //Record into global memory, here, namely input pointer.
    input[tid] = sharedMem[tid]; input[(num/2) + tid] = sharedMem[(num/2) + tid];
    //Synchronise the thread block to ensure data is consistent at the end in the global memory.
    tb.sync();
    
}

__global__ void mulInPar(float2* x, float2* y, float2* result){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    result[tid] = mulFloat2(x[tid], y[tid]);
}

void multiplyPolCoeffs(float2* poly1, float2* poly2, float2* result, int N){
    float2* polyReversed;
    polyReversed = (float2*)malloc(N * sizeof(float2));

    //poly1
    //Bit reverse, update later on.
    for(int i = 0; i < N; i++){
        polyReversed[bitReverse(i, N)] = poly1[i];
    }
    
    float2* d_vecPoly1;
    cudaMalloc(&d_vecPoly1, N * sizeof(float2));

	cudaMemcpy(d_vecPoly1, polyReversed, N * sizeof(float2), cudaMemcpyHostToDevice);

    //Forward fft
    parButterFlies<<< 1, N/2, N * sizeof(float2) >>>(N, d_vecPoly1, true);
    cudaDeviceSynchronize();

    /////////////////

    //poly2
    //Bit reverse, update later on.
    float2* d_vecPoly2;
    cudaMalloc(&d_vecPoly2, N * sizeof(float2));

    for(int i = 0; i < N; i++){
        polyReversed[bitReverse(i, N)] = poly2[i];
    }

	cudaMemcpy(d_vecPoly2, polyReversed, N * sizeof(float2), cudaMemcpyHostToDevice);

    //Forward fft
    parButterFlies<<< 1, N/2, N * sizeof(float2) >>>(N, d_vecPoly2, true);
    cudaDeviceSynchronize();

    ///////////////

    //Now that we have our results of fft at device pointer d_vecPoly1 and d_vecPoly2
    //We apply multiplication on them.
    float2* d_resultVecPoly;
    cudaMalloc(&d_resultVecPoly, N * sizeof(float2));

    //TODO better thread block count
    mulInPar<<< 1 , N >>>(d_vecPoly1, d_vecPoly2, d_resultVecPoly);
    cudaDeviceSynchronize();

    //Pass d_resultVecPoly to host for bit reversal, then back into device for inverse fft.
    cudaMemcpy(poly1, d_resultVecPoly, N * sizeof(float2), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++){
        polyReversed[bitReverse(i, N)] = poly1[i];
    }

    cudaMemcpy(d_resultVecPoly, polyReversed, N * sizeof(float2), cudaMemcpyHostToDevice);

    parButterFlies<<<1, N/2, N * sizeof(float2)>>>(N, d_resultVecPoly, false);
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_resultVecPoly, N * sizeof(float2), cudaMemcpyDeviceToHost);

    cudaFree(d_resultVecPoly);
    cudaFree(d_vecPoly1);
    cudaFree(d_vecPoly2);
    free(polyReversed);
}