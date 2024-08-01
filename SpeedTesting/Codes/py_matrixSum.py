import numpy as np

def test(a, b, len):
	result = np.zeros(len, dtype="int32")

	for i in range(len):
		result[i] = a[i] + b[i] 

	#return result

	