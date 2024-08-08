import numpy as np
import timeit #Used to measure the time of the function executions

import wrappedCuda			#Cuda --> (wrappedCuda shared library name specified in setup_cuda.py)

#Create randomly filled arrays
len = 2 ** 25
#Note that performance of cuda is present when len is large for summation

def refreshVec():
	global a, b
	a = np.array(np.random.randint(2048, size=len), dtype=np.int32)
	b = np.array(np.random.randint(2048, size=len), dtype=np.int32)

test_code = """
from __main__ import helperCuda, refreshVec
refreshVec()
"""

#Helper functions to call functions wrapped
#Used for the healthy usage of timeit class below

def helperCuda():
	#Cython helper --> Wrapped Cuda code
	wrappedCuda.test0(a, b, len)

#stmt --> Defines the statement to be executed
#setup --> Before timing the operation, we specify things to be done priorly
#number --> Number of executions to run the stmt

#(1)

total = 0;

repetition = 100
for i in range(repetition):
	cudaV = timeit.timeit(stmt="helperCuda()",
						setup=test_code,
						number=1)
	
	total += cudaV #Add total time

print("\nRepeating the function {} time, below are the execution times:\n".format(repetition))
print("Time it took for -cuda wrapped with cython- is (sec): {}".format(total))
