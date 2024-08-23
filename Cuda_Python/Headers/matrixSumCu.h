#ifndef MATRIXSUMCU_H
#define MATRIXSUMCU_H

//This header is the header for ../Codes/matrixSum.cu

#include <iostream>
#include <vector>

void calcSumPar(int* a, int* b, int* result, int len);
void calcSum(int* a, int* b, int* result, int len);
void helperPar(int* a, int* b, int* result, int len);
int* allocate_on_heap(size_t size);
void initialize();

#endif