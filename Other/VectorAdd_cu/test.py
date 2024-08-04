import numpy as np
from wrappedCuda import vector_add

#a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
#b = np.array([10, 20, 30, 40, 55], dtype=np.int32)

a = np.array(np.random.randint(100, size=2**10), dtype=np.int32)
b = np.array(np.random.randint(100, size=2**10), dtype=np.int32)

vector_add(a, b)


