import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "vectorAdd.h":
	void calcPar(const int *a,
             	 const int *b,
             	 int *c,
             	 int N)

def vector_add(arr1, arr2):

	#Transform arrays into numpy array --> np.ndarray (type) using np.array (method)
	cdef np.ndarray[np.int32_t] array1 = np.array(arr1, dtype="int32")
	cdef np.ndarray[np.int32_t] array2 = np.array(arr2, dtype="int32")

	assert array1.shape[0] == array2.shape[0], "Dimensions of given arrays do not match"
	cdef int N = array1.shape[0]

	cdef int *resultArr
	resultArr = <int*>malloc(N * sizeof(int))

	calcPar(<int*>array1.data, <int*>array2.data,
							 resultArr, N)

	#Free the allocated memory
	free(resultArr)




	

