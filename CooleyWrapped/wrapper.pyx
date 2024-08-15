import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "cuda_runtime.h":
    ctypedef struct float2:
        float x
        float y

cdef extern from "Base2.cuh":
    float2* prepOperations(float2* input, int N)

def performOp_Cy(np.ndarray coeff, int N):
    cdef float2* data_ptr = <float2*>coeff.data

    for i in range(0, N):
        print(data_ptr[i])

    #In function below, do fft and apply inverse fft.
    prepOperations(data_ptr, N);

    #Return the result.
    return coeff






