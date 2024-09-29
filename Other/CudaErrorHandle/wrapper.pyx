"""
Wrapper for cuda 
"""

cimport cython

cdef extern from "ErrorSpecifier.h":
    #Three types of tries
    int libFunction(int x)
    const char* libFunctionC(int x)
    ###cudaError_t libFunctionE(int x) --> Take a look how we may wrap

    #Global error checker located inside cuda
    ###const char* cudaLibGetErrorString(cudaError_t errorType) --> Take a look how we may wrap
    const char* cudaLibGetErrorString(int errorType)

    #Tryout for cudaError thrown
    void throwError()

@cython.boundscheck(False)  # Disable bounds checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def stringReturned(int x):
    return libFunctionC(x).decode('utf-8') # Decoding C string to Python string

def throwCudaError():
    throwError()
    #This function terminates and does not proceeds with the remaining python code.

@cython.boundscheck(False) 
@cython.wraparound(False)
def intReturned(int x):
    #Without doing anything other than calling the function in python, we can see the error
    cdef int errorNo = libFunction(x)
    str = errorChecker(errorNo) #Get the string of the error
    print(str.decode('utf-8')) #print the error

    return errorNo #Return the error id 


cdef errorChecker(int errorNo):
    if(errorNo > 0):
        pass # Decide on what to do in this case, cudaError's are identified with errorNo >= 0 with 0 being cudaSuccess
    elif(errorNo < 0):
        return cudaLibGetErrorString(errorNo) #Meaning we have a library related problem
    else:
        pass #This means succesful so do not do anything


        




	