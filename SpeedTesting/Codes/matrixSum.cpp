#include <vector>
#include "../Headers/matrixSumCpp.h"

void calcSum(int* a, int* b, int* result, int len){
	for(int i = 0; i < len; i++){
		result[i] = a[i] + b[i];
	}
}