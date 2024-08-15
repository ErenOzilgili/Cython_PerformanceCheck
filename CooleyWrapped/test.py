import numpy as np
from tfhe_fft import performOp_Cy

# Define a dtype that matches float2
float2_dtype = np.dtype([('x', np.float32), ('y', np.float32)])

# Create an array of float2
arr = np.array((2 ** 4) *[(2.0, 0.0), (4.0, 0.0), (-1.0, 0.0), (5.0, 0.0)], dtype=float2_dtype)

coeffToGet = performOp_Cy(arr, 2 ** 6)

print("After doing fft and inverse fft, we have these coefficients:")
print(coeffToGet)

