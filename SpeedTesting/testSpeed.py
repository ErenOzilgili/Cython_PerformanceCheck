import numpy as np
import timeit #Used to measure the time of the function executions

#Importing functions to call
import Codes.py_matrixSum	#Python
import wrappedCyC			#C and Cython --> They are in the same shared lib, (wrappedCyC shared library name specified in setup_cuda.py)
import wrappedCuda			#Cuda --> (wrappedCuda shared library name specified in setup_cuda.py)

#Create randomly filled arrays
len = 2 ** 30 
#Note that performance of cuda is present when len is large for summation
a = np.array(np.random.randint(100, size=len), dtype=np.int32)
b = np.array(np.random.randint(100, size=len), dtype=np.int32)

#Helper functions to call functions wrapped
#Used for the healthy usage of timeit class below
def helperPy():
	#Python helper
	Codes.py_matrixSum.test(a, b, len)

def helperCy():
	#Cython helper --> Usual Cython
	wrappedCyC.test0(a, b, len)

def helperC():
	#Cython helper --> Wrapped C code
	wrappedCyC.test1(a, b, len)

def helperCuda():
	#Cython helper --> Wrapped Cuda code
	wrappedCuda.test0(a, b, len)

#stmt --> Defines the statement to be executed
#setup --> Before timing the operation, we specify things to be done priorly
#number --> Number of executions to run the stmt

#(1)
repetition = 1
pythonV = timeit.timeit(stmt="helperPy()",
						setup="from __main__ import helperPy",
						number=repetition)

cythonV = timeit.timeit(stmt="helperCy()",
						setup="from __main__ import helperCy",
						number=repetition)

cV = timeit.timeit(stmt="helperC()",
						setup="from __main__ import helperC",
						number=repetition)

cudaV = timeit.timeit(stmt="helperCuda()",
					  setup="from __main__ import helperCuda",
					  number=repetition)

"""	
	Same matrix sum calculated on 
	python ,cython ,c ,and cuda.
	Time each function took and time relative
	to python are printed in terminal below
"""

print("\nRepeating the functions {} time, below are the execution times:\n".format(repetition))
print("Time it took for -python- is: {}".format(pythonV))	
print("Time it took for -cython- is: {}".format(cythonV))
print("Time it took for -c wrapped with cython- is: {}".format(cV))
print("Time it took for -cuda wrapped with cython- is: {}".format(cudaV))
print("\n-Cython- is {} times faster than -python-".format(pythonV / cythonV))
print("\n-C- is {} times faster than -python-".format(pythonV / cV),"\n")
print("\n-Cuda- is {} times faster than -python-".format(pythonV / cudaV),"\n")

"""
Below can be used to not calculate python
function time, which takes much time.
So, (2) can be used to test the same things
instead of the above (1).
For comparing cython, c ,and cuda
"""

"""
#(2)
repetition = 1
cythonV = timeit.timeit(stmt="helperCy()",
							setup="from __main__ import helperCy",
							number=repetition)

cV = timeit.timeit(stmt="helperC()",
							setup="from __main__ import helperC",
							number=repetition)

cudaV = timeit.timeit(stmt="helperCuda()",
							setup="from __main__ import helperCuda",
							number=repetition)

print("\nRepeating the functions {} time, below are the execution times:\n".format(repetition))
print("Time it took for -cython- is: {}".format(cythonV))
print("Time it took for -c wrapped with cython- is: {}".format(cV))
print("Time it took for -cuda wrapped with cython- is: {}".format(cudaV))
print("\n-C- is {} times faster than -cython-".format(cythonV / cV))
print("\n-Cuda- is {} times faster than -cython-".format(cythonV / cudaV),"\n")
"""