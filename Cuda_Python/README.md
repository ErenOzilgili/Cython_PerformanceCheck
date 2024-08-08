##How To Run
--
From the ./Cuda_Python folder run these commands:

(1) To run the speed test for Cuda in terminal

      nvcc ./Codes/matrixSum.cu -o test
      ./test
      
(2) To run the speed test for python

    python setup_cuda.py build_ext --inplace
    python testSpeed.py

##Note
--
Vector sum is calculated on the vectors of length 2^25.




