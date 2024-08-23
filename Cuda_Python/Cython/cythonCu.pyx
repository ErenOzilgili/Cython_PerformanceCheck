"""
Wrapper for cuda 
"""
import ctypes
import numpy as np
cimport numpy as np
cimport cython

from cython cimport boundscheck, wraparound
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy as c_memcpy
from cpython cimport array
import array
from libc.stdlib cimport malloc, free

cdef extern from "../Headers/matrixSumCu.h":
    void calcSumPar(int* a, int* b, int* result, int len)
    void calcSum(int* a, int* b, int* result, int len)
    void helperPar(int* a, int* b, int* result, int len)
    int* allocate_on_heap(size_t size)
    #void initialize()

"""
cdef process_gpuArr(int* gpu_ptr_a, int* gpu_ptr_b, gpu_ptr_result, int len):
    pass
"""

@boundscheck(False)
@wraparound(False)
def helpPar(a, b, result, int len):
    cdef int[::1] a_view = a
    cdef int[::1] b_view = b
    cdef int[::1] result_view = result

    helperPar(&a_view[0], &b_view[0], &result_view[0], len)

def calcSumParPtrs(np.ndarray[np.int32_t, ndim=1] a,
                   np.ndarray[np.int32_t, ndim=1] b,
                   np.ndarray[np.int32_t, ndim=1] result,
                   int len):
    # Size of the array
    cdef size_t size = len * sizeof(int)

    cdef int* a_ptr = allocate_on_heap(size)
    cdef int* b_ptr = allocate_on_heap(size)
    cdef int* result_ptr = allocate_on_heap(size)

    # Copy data from numpy array to new memory
    c_memcpy(a_ptr, <int*>a.data, size)
    c_memcpy(b_ptr, <int*>b.data, size)
    c_memcpy(result_ptr, <int*>result.data, size)
    

    """
    a_ptr = <int*>a.data
    b_ptr = <int*>b.data
    result_ptr = <int*>result.data
    """

    calcSum(a_ptr, b_ptr, result_ptr, len)

@boundscheck(False)
@wraparound(False)
def simpleSum(a, b, result, int len):
    cdef int[::1] a_view = a
    cdef int[::1] b_view = b
    cdef int[::1] result_view = result

    calcSum(&a_view[0], &b_view[0], &result_view[0], len)

"""
@boundscheck(False)
@wraparound(False)
def testPtr(int* a, int* b, int* c, len):
    cdef int* a_ptr = <int*>a
    cdef int* b_ptr = <int*>b
    cdef int* c_ptr = <int*>c

    calcSumPar(a_ptr, b_ptr, c_ptr, len)
"""

#Calling C++
@cython.boundscheck(False)  # Disable bounds checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def test0(a, b, result, int len):
    # Working with numpy arrays
    """
    from cpython cimport view
    view.array ????
    """

    # 0.014487740899858182 (amortized sec)
    #cdef np.ndarray[np.int32_t, ndim=1] result = np.zeros(len, dtype=np.int32)
    cdef int[::1] a_view = a
    cdef int[::1] b_view = b
    cdef int[::1] result_view = result

    #Here we call the c++ function calcSum --> in Codes folder matrixSum.cpp
    calcSumPar(&a_view[0], &b_view[0], &result_view[0], len)

    """
    #Working with python arrays
    # 0.017987080499995044 (amortized sec)
    cdef array.array a_arr = a

    cdef array.array b_arr = b

    cdef array.array result_arr = result

    calcSumPar(a_arr.data.as_ints, b_arr.data.as_ints, result_arr.data.as_ints, len)
    """

    """
    cdef int* a_ptr = <int*> a_arr.data.as_voidptr()
    cdef int* b_ptr = <int*> b_arr.data.as_voidptr()
    cdef int* result_ptr = <int*> result_arr.data.as_voidptr()
    """

    # Working with lists
    """
    # 0.13941950340013137 (amortized sec) no return
    cdef np.ndarray[np.int32_t, ndim=1] result = np.zeros(len, dtype=np.int32)
    cdef int[:] a_view = np.array(a, dtype=np.int32)
    cdef int[:] b_view = np.array(b, dtype=np.int32)
    cdef int[:] result_view = result
    """

    """
    # 0.15665736029986874 (amortized sec) no return
    cdef array.array a_arr = array.array('i', a)
    cdef int[::1] a_view = a_arr

    cdef array.array b_arr = array.array('i', b)
    cdef int[::1] b_view = b_arr

    result = len * [0]
    cdef array.array c_arr = array.array('i', result)
    cdef int[::1] result_view = c_arr
    """

    """
    # Result parameter is passed to the function
    # 0.14535312489897478 (amortized sec)
    cdef array.array a_arr = array.array('i', a)
    cdef int[::1] a_view = a_arr

    cdef array.array b_arr = array.array('i', b)
    cdef int[::1] b_view = b_arr

    cdef array.array c_arr = array.array('i', result)
    cdef int[::1] result_view = c_arr
    """

    """
    # 0.03135294900057488 (amortized sec) no return
    cdef int* a_view = <int*> malloc(len * sizeof(int))
    cdef int* b_view = <int*> malloc(len * sizeof(int))
    cdef int* result_view = <int*> malloc(len * sizeof(int))

    cdef int i = 0
    for i in range(len):
        a_view[i] = a[i]
        b_view[i] = b[i]
    """    


    """
    i = 0
    result = [result_view[i] for i in range(len)]

    return result
    """
	
