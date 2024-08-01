"""
Wrapper for C++ --> C++ function wrapped inside test1 function

Also includes cython only function --> test0
"""

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "../Headers/matrixSumCpp.h":
	void calcSum(int* a, int* b, int* result, int len)

#Only cython
@cython.boundscheck(False)  # Disable bounds checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def test0(a, b, int len):
	cdef int i
	cdef int[:] x = a
	cdef int[:] y = b
	cdef int[:] result = np.zeros(len, dtype="int32")

	for i in range(len):
		result[i] = x[i] + y[i]

	return result

#Calling C++
@cython.boundscheck(False)  # Disable bounds checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def test1(a, b, int len):
	cdef np.ndarray[np.int32_t, ndim=1] result = np.zeros(len, dtype=np.int32)
	cdef int[:] a_view = a
	cdef int[:] b_view = b
	cdef int[:] result_view = result
    
    #Here we call the c++ function calcSum --> in Codes folder matrixSum.cpp
	calcSum(&a_view[0], &b_view[0], &result_view[0], len)

	return result