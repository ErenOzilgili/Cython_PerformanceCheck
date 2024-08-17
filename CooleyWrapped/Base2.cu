#include <iostream>
#include <stdio.h>
#include <cooperative_groups.h>
#include <complex>
#include <cmath>
#include "Base2.cuh"

using std::cout, std::endl;

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
    //Will have threads = num / 2.
    int logNum = __float2int_rz(log2f((float)num));  // Using log2f for single-precision log2
    int offset = 1;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int currTid = tid;

    if(coeffToVec){
        for(int i = 0; i < logNum; i++){
            currTid = tid + ((tid / offset) * offset);

            // Calculate the exponent and twiddle factor
            float angle = (tid % offset) * (-2.0f) * M_PI / (2 * offset);
            float2 rootCom = make_float2(cosf(angle), sinf(angle)); 

            // Perform the butterfly operations
            cooleyButterOp(input[currTid], input[currTid + offset], rootCom);

            offset *= 2;
            __syncthreads();
        }
    }
    else{
        //printf("Implement inverse fft for here, vector to coeff\n");
        for(int i = 0; i < logNum; i++){
            currTid = tid + ((tid / offset) * offset);

            // Calculate the exponent and twiddle factor
            float angle = (tid % offset) * (2.0f) * M_PI / (2 * offset);
            float2 rootCom = make_float2(cosf(angle), sinf(angle)); 

            // Perform the butterfly operations
            cooleyButterOp(input[currTid], input[currTid + offset], rootCom);

            offset *= 2;
            __syncthreads();
        }
        //Divide by factor of n. Notice that offset is n by the end of the loop.
        invCooleyNormalize(tid, offset/2, input); //So we need to pass the half of the current offset.
        
    }
}

__global__ void mulInPar(float2* x, float2* y, float2* result){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    result[tid] = mulFloat2(x[tid], y[tid]);
}

void multiplyPolCoeffs(float2* poly1, float2* poly2, float2* result, int N){
    //float2* result;
    //result = (float2*)malloc(N * sizeof(float2));

    float2* polyReversed;
    polyReversed = (float2*)malloc(N * sizeof(float2));

    //poly1
    //Bit reverse, update later on.
    for(int i = 0; i < N; i++){
        polyReversed[bitReverse(i, N)] = poly1[i];
    }

    //Print, later on delete
    for(int i = 0; i < N; i++){
        cout << "polyReversed1 " << i << "th value:" << polyReversed[i].x << " " << polyReversed[i].y << endl;
    }

    float2* d_vecPoly1;
    cudaMalloc(&d_vecPoly1, N * sizeof(float2));

	cudaMemcpy(d_vecPoly1, polyReversed, N * sizeof(float2), cudaMemcpyHostToDevice);

    cout << "1" << endl;

    //Forward fft
    parButterFlies<<< 1, N/2 >>>(N, d_vecPoly1, true);
    cudaDeviceSynchronize();

    cout << "2" << endl;

    /////////////////

    //poly2
    //Bit reverse, update later on.
    float2* d_vecPoly2;
    cudaMalloc(&d_vecPoly2, N * sizeof(float2));

    cout << "3" << endl;

    for(int i = 0; i < N; i++){
        polyReversed[bitReverse(i, N)] = poly2[i];
    }

    //Print, later on delete
    for(int i = 0; i < N; i++){
        cout << "polyReversed2 " << i << "th value:" << polyReversed[i].x << " " << polyReversed[i].y << endl;
    }


	cudaMemcpy(d_vecPoly2, polyReversed, N * sizeof(float2), cudaMemcpyHostToDevice);

    cout << "4" << endl;

    //Forward fft
    parButterFlies<<< 1, N/2 >>>(N, d_vecPoly2, true);
    cudaDeviceSynchronize();

    cout << "5" << endl;

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

    parButterFlies<<<1, N/2>>>(N, d_resultVecPoly, false);
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_resultVecPoly, N * sizeof(float2), cudaMemcpyDeviceToHost);

        //Print, later on delete
    for(int i = 0; i < N; i++){
        cout << "result " << i << "th value:" << result[i].x << " " << result[i].y << endl;
    }

    cudaFree(d_resultVecPoly);
    cudaFree(d_vecPoly1);
    cudaFree(d_vecPoly2);
    free(polyReversed);

    //return result;
}

//Helper function for operating
//fft and inverse fft with the data
//coming from the python.
void prepOperations(float2* input, int N){
    //Hold the reversed bits of coefficients in host side.
    float2* inputReversed;
    inputReversed = (float2*)malloc(N * sizeof(float2));

    //Bit reverse, update later on.
    for(int i = 0; i < N; i++){
        inputReversed[bitReverse(i, N)] = input[i];
    }

    float2* device_vecs;
    cudaMalloc(&device_vecs, N * sizeof(float2));

	cudaMemcpy(device_vecs, inputReversed, N * sizeof(float2), cudaMemcpyHostToDevice);

    //Forward fft
    parButterFlies<<< 1, N/2 >>>(N, device_vecs, true);
    cudaDeviceSynchronize();

    /////////////////////////////////////////

    //For holding the fft'ed vector in device, will be used to store bit reversed version.
    float2* device_vecsReversed;
    cudaMalloc(&device_vecsReversed, N * sizeof(float2));

    // (***)
    cudaMemcpy(input, device_vecs, N * sizeof(float2), cudaMemcpyDeviceToHost);
    //Bit reverse, update later on.
    for(int i = 0; i < N; i++){
        inputReversed[bitReverse(i, N)] = input[i];
    }
    cudaMemcpy(device_vecsReversed, inputReversed, N * sizeof(float2), cudaMemcpyHostToDevice);

    /* (***)
    //Bit reverse again for doing inverse fft, update later on.
    for(int i = 0; i < N; i++){
        device_vecsReversed[bitReverse(i, N)] = device_vecs[i];
    }
    cout << "7" << endl;
    */

    //Delete later on
    cout << "After fft is performed and result is returned back to host, this is the result." << endl;
    for(int i = 0; i < N; i++){
        printf("%dth element: %f , %f\n", i , input[i].x ,  input[i].y);
    }
    cout << "After above, it will be bit reversed and recorded into inputReversed." << endl;
    cout << "Then it will be recorded into device_vecsReversed, will apply inverse fft and back into host input." << endl;

    //Inverse fft
    parButterFlies<<< 1, N/2 >>>(N, device_vecsReversed, false);
    cudaDeviceSynchronize();

    //Copy back into the input, the original results.
    cudaMemcpy(input, device_vecsReversed, N * sizeof(float2), cudaMemcpyDeviceToHost);

    //Free not needed elements
    cudaFree(device_vecs);
    cudaFree(device_vecsReversed);
    free(inputReversed);
}

/*
int main(){
    int arr[8] = {0, 1, 2 ,3, 4, 5, 6, 7};

    int arrRev[8];

    for(int i = 0; i < 8 ; i++){
        arrRev[bitReverse(i, 8)] = arr[i];
    }

    for(int j = 0; j < 8; j++){
        cout << arrRev[j] << " ";
    }
}
*/