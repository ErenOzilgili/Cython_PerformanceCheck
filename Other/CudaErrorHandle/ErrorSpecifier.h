#ifndef ERRORSPECIFIER_H
#define ERRORSPECIFIER_H

#include <cuda_runtime.h>

//Three types of tries
int libFunction(int x);
cudaError_t libFunctionE(int x);
const char* libFunctionC(int x);

//Global error checker located inside cuda
const char* cudaLibGetErrorString(cudaError_t errorType);
const char* cudaLibGetErrorString(int errorType);

//Tryout for cudaError thrown
void throwError();

#endif