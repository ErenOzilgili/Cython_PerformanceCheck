import array
import numpy as np
import timeit #Used to measure the time of the function executions
import wrappedCuda
import ctypes

#import pycuda.driver as cuda
import pycuda.autoinit
"""
import pycuda.gpuarray as gpuarray
"""

#Create randomly filled arrays
len = 2 ** 20
#Note that performance of cuda is present when len is large for summation

def refreshVec():
    global a, b, a_ptr, b_ptr, result_ptr, result, gpu_array_a, gpu_array_b, gpu_array_result
    #a = np.array(np.random.randint(2048, size=len), dtype=np.int32)
    #b = np.array(np.random.randint(2048, size=len), dtype=np.int32)

    a = np.array(np.random.randint(2048, size=len), dtype=np.int32)
    b = np.array(np.random.randint(2048, size=len), dtype=np.int32)
    result = np.zeros(len, dtype=np.int32)

    """
    gpu_array_a = gpuarray.Array(shape=, )
    gpu_array_b = gpuarray.Array()
    gpu_array_result = gpuarray.to_gpu(result)
    """

    """
    result = [0 for i in range(len)]
    a = [np.random.randint(2048) for i in range(len)]
    b = [np.random.randint(2048) for i in range(len)]
    """

    """
    a = np.array(np.random.randint(2048, size=len), dtype=np.int32)
    b = np.array(np.random.randint(2048, size=len), dtype=np.int32)
    """

    """
    a = np.array(np.random.randint(2048, size=len), dtype=np.int32)
    b = np.array(np.random.randint(2048, size=len), dtype=np.int32)
    result = np.zeros(len, dtype=np.int32)
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    """

    #ctypes.Array(ctypes.c_double, len(m.data))(*m.data)
    
    """
    a = array.array('i', a)
    b = array.array('i', b)
    result = array.array('i', result)
    """
    
test_code = """
from __main__ import helperCuda, refreshVec, resetGpuDev
refreshVec()
"""

#Helper functions to call functions wrapped
#Used for the healthy usage of timeit class below

def resetGpuDev():
	wrappedCuda.resetDevice()

def helperCuda():
    #Pass gpu pointers
    #wrappedCuda.process_gpuArr(gpu_array_a.gpudata, gpu_array_b.gpudata, gpu_array_result.gpudata, len)

    
	#Cython helper --> Wrapped Cuda code
    #Paralel kernel
    wrappedCuda.test0(a, b, result, len)
    #wrappedCuda.calcSumParPtrs(a, b, result, len)

    #ptr test
    #wrappedCuda.testPtr(a_ptr, b_ptr, result_ptr, len)

    #Normal c sum
    #wrappedCuda.simpleSum(a, b, result, len)

    #Helper for kernel
    #wrappedCuda.helpPar(a, b, result, len)

#stmt --> Defines the statement to be executed
#setup --> Before timing the operation, we specify things to be done priorly
#number --> Number of executions to run the stmt

total = 0

repetition = 10
for i in range(0, repetition):
    cudaV = timeit.timeit(setup=test_code,
                          stmt="helperCuda()",
                        number=1)

    total += cudaV #Add total time

print("\nRepeating the function {} time, below are the execution times:\n".format(repetition))
print("Time it took for -cuda wrapped with cython- is (sec) (amortized): {}".format(total/repetition))