from wrapper import *

#Return string
print(stringReturned(10)) # noError --> string assigned to cudaSuccess in Cuda internally

#We will be printing the error associated with the problem inside cython using errorChecker function
#Maybe we can use the returned integer for giving user another try at this step
errorType = intReturned(-10) 

#Return cudaError, currently not resolved

throwCudaError() #Here we encounter an cuda runtime error so below wont be printed
print("Won't be printed")

