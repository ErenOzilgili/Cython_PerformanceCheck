import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "cuda_runtime.h":
    ctypedef struct float2:
        float x
        float y

cdef extern from "Base2.cuh":
    void prepOperations(float2* input, int N)
    void multiplyPolCoeffs(float2* poly1, float2* poly2, float2* result, int N)

def mulPolys(np.ndarray coeff1, np.ndarray coeff2, np.ndarray empty, int N):
    cdef float2* result = <float2*>empty.data
    cdef float2* data1_ptr = <float2*>coeff1.data
    cdef float2* data2_ptr = <float2*>coeff2.data

    """
    print("First coeffs:")
    for i in range(0, N):
        print(data1_ptr[i])
    """

    """
    print("Second coeffs:")
    for i in range(0, N):
        print(data2_ptr[i])
    """

    multiplyPolCoeffs(data1_ptr, data2_ptr, result, N)
    #result = multiplyPolCoeffs(data1_ptr, data2_ptr, N)

    """
    print("Result")
    for i in range(0, N):
        print(result[i])
    """

    """
    print("Empty")
    for i in range(0, N):
        print(empty[i])
    """

    return empty

def performOp_Cy(np.ndarray coeff, int N):
    cdef float2* data_ptr = <float2*>coeff.data

    for i in range(0, N):
        print(data_ptr[i])

    #In function below, do fft and apply inverse fft.
    prepOperations(data_ptr, N);

    #Return the result.
    return coeff

def schoolBookMul(np.ndarray coeff1,
                  np.ndarray coeff2,
                  np.ndarray empty, int N):
    #Here, N is the length of the coeff1 and coeff2
    cdef int iterOut = 0
    cdef int iterIn = 0

    cdef float x = 0
    cdef float y = 0

    for iterOut in range(0, N):
        record = (0, 0)
        for iterIn in range(0, N):
            x += coeff1[iterOut]['x'] * coeff2[iterIn]['x'] - coeff1[iterOut]['y'] * coeff2[iterIn]['y']
            y += coeff1[iterOut]['x'] * coeff2[iterIn]['y'] + coeff1[iterOut]['y'] * coeff2[iterIn]['x']
        empty[iterOut + iterIn]['x'] = x 
        empty[iterOut + iterIn]['y'] = y

    return empty