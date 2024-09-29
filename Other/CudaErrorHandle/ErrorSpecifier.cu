//By including this we can reach cudaError enum.
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

const char* cudaLibGetErrorString(cudaError_t errorType);
const char* cudaLibGetErrorString(int errorType);

/*
Here, errors defined related to library.
*/
typedef enum libraryErrors{
    LIB_ERROR_1 = -1,
    LIB_ERROR_2 = -2,
    LIB_ERROR_3 = -3,
    LIB_ERROR_4 = -4
}libError_t;

//(1)
//Experimenting if a function call cudaError as int since they
//  are declared as ints in the cudaError enum defined in cuda_runtime.h
int funcErrorReturn(){
    return cudaSuccess;
}
int libFunction(int x) {
    
    //Checking for invalid input for example
    if (x < 0) {
        return LIB_ERROR_1;  // Return custom error
    }

    //Success in case everything is fine
    return cudaSuccess;
}

//(2)
//Returning cudaError
//Let the below function be a sample library function
cudaError_t libFunctionE(int x) {
    
    //Checking for invalid input for example
    if (x < 0) {
        return (cudaError_t)LIB_ERROR_1;  // Return custom error
    }

    //Success in case everything is fine
    return cudaSuccess;
}

const char* libFunctionC(int x) {
    //Checking for invalid input for example
    if (x < 0) {
        return cudaLibGetErrorString((cudaError_t)LIB_ERROR_1);  // Return custom error
    }

    //Success in case everything is fine
    return cudaLibGetErrorString(cudaSuccess);
}

const char* cudaLibGetErrorString(cudaError_t errorType){
    /*
    Note that default cases are basic cudaError defined in the CUDA Runtime API.
    Other cases are library defined errors for user to obey or similar things.
    */
    int error = errorType;

    switch (error) {
        case LIB_ERROR_1:
            return "Library Error 1: Integer given to function can't be negative.";
        case LIB_ERROR_2:
            return "Library Error 2: Too many inputs.";
        case LIB_ERROR_3:
            return "Library Error 3: Something went wrong.";
        case LIB_ERROR_4:
            return "Library Error 4: Another issue occurred.";
        default:
            return cudaGetErrorString(errorType);  // Return CUDA error string if it's a CUDA error
    }
}

const char* cudaLibGetErrorString(int errorType){
    /*
    Note that default cases are basic cudaError defined in the CUDA Runtime API.
    Other cases are library defined errors for user to obey or similar things.
    */

    switch (errorType) {
        case LIB_ERROR_1:
            return "Library Error 1: Integer given to function can't be negative.";
        case LIB_ERROR_2:
            return "Library Error 2: Too many inputs.";
        case LIB_ERROR_3:
            return "Library Error 3: Something went wrong.";
        case LIB_ERROR_4:
            return "Library Error 4: Another issue occurred.";
        default:
            return cudaGetErrorString((cudaError_t)errorType);  // Return CUDA error string if it's a CUDA error
    }
}

void throwError(){
    throw cudaErrorInvalidValue;
}

int main(){
    //Returning int as error (1) works.
    cout << funcErrorReturn() << endl;

    //Just a way to reach enum type (outputs -1)
    cout << libError_t::LIB_ERROR_1 << "\n" <<endl; 

    //(2) We return cudaError_t
    cudaError_t errorNo;

    errorNo = libFunctionE(10);//With cudaError_t
    cout << cudaLibGetErrorString(errorNo) << endl; 
    cout << libFunction(10) << "\n" << endl;//With integer

    errorNo = libFunctionE(-10);//With cudaError_t
    cout << cudaLibGetErrorString(errorNo) << endl;
    cout << libFunction(-10) << "\n" <<endl;//With integer

    //Checking if we can result in default case in the function cudaLibGetErrorString.
    //This will be printed which is a CUDA-Runtime error type defined in cudaError enum internally. 
    printf("Error: %s\n", cudaLibGetErrorString(cudaErrorInvalidValue));
    cout << "\n";

    cout << (cudaLibGetErrorString(16)) << "\n";

    return 0;
}
