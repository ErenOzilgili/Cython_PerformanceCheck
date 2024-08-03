# Cython_PerformanceCheck -Overview-
---
A library with examples to show the performance gains due to usage of cython. It will be used to demonstrate how and why efficiency increases when used cython. SpeedTesting is a folder for speed comparisons. 


## Sample Run
---
After dependencies are satisfied, simply cd into (SpeedTesting) folder. 

    cd SpeedTesting

Afterwards, running the following commands will help you
run the sample code (testSpeed.py).

    python setup_cy.py build_ext --inplace

    python setup_cuda.py build_ext --inplace

    python testSpeed.py 
  

Below is a sample run:

    Repeating the functions 1 time, below are the execution times:
    
    Time it took for -python- is: 451.4805138280026
    Time it took for -cython- is: 4.223426950997236
    Time it took for -c wrapped with cython- is: 3.91828661800173
    Time it took for -cuda wrapped with cython- is: 1.8142729489991325
    
    -Cython- is 106.89909380850044 times faster than -python-
    
    -C- is 115.22396339098113 times faster than -python- 

    -Cuda- is 248.84927820649463 times faster than -python- 
    

## Dependencies
---


## Cython Speed
---
--> First of all, 

	python is an interpreted language, which means that python code won't be directly 
 	compiled into machine code. On the other hand, cython will compile the python into
  	C code, then it will be compiled into machine code. Resulting machine code will 
   	run faster than the interpreted code.

--> Python is dynamically typed, 

	which means that variable types are determined and checked at runtime rather than 
 	during compilation. On the other hand, Cython allows for static typing, meaning 
  	you can declare the types of variables explicitly. This feature of cython prevents
   	the overhead associated with dynamic typing in Python, leading to faster execution.

--> Since Cython code is closer to C, 

	it has less overhead than Python. This includes reduced function call overhead, 
 	memory management optimizations, and more efficient looping constructs.

--> Cython can use C-level memory management, 

	which can be more efficient than Python’s garbage collector, especially for large 
 	datasets or performance-critical applications.

--> Cython seamlessly integrates with C and C++ libraries, 

	allowing for the use of highly optimized external libraries directly 
 	in the code, which can enhance performance.


## Syntax
---
--> Method Declerations

	    Methods can be declared as cdef, cpdef, or def:
	    cdef: Only accessible from Cython, not from Python.
	    cpdef: Generates both a Cython-accessible and a Python-accessible method, providing flexibility with a slight overhead.
	    def: Standard Python method, accessible from Python but less efficient.

--> More On 'cdef' Keyword

	In Cython, the cdef keyword is used to declare C variables, types, and functions. However, Cython also allows you to declare Python variables 		with cdef to achieve more efficient attribute access and memory management. This is why cdef list x is valid syntax in Cython: it allows you 		to declare a variable x as a Python list, with some performance advantages over regular Python variable declarations.

--> For importing methods, classes from c++, cdef extren from "___" is used. Here is an example

    '
    cdef extern from "matrixSum.h":
	    void calcSum(int* a, int* b, int* result, int len)
    '
    This block of code allows cython to recognizes the methods, classes and other structures declared inside extern block and
    allows its usage.
    
    
## Data Conversions
---
--> Python: int

    C Type: cdef int or cdef long
    Cython Conversion: Python int maps to int or long in C. Use cdef int for most integer operations.

--> Python: float

    C Type: cdef float or cdef double
    Cython Conversion: Python float is typically mapped to double in C. Use cdef float for single precision or cdef double for double precision.

--> Python: bool

    C Type: cdef bint
    Cython Conversion: Python bool maps to bint in Cython, which is a type-safe boolean representation.

--> NumPy Arrays

    Cython Type: cdef np.ndarray[args] for one-dimensional arrays
    Conversion: Convert NumPy arrays to Cython memoryviews (type[:])

--> Python: list

    C Type: cdef int[:] (or appropriate [:] for other types) (memoryviews)
    Cython Conversion: Lists can be converted to memoryviews if they are NumPy arrays. For example, np.array(list, dtype=np.int32) can be used, and             then cdef int[:] x can be used to access it.
    Why memoryviews? --> Using memoryviews in Cython is often the most efficient way to handle lists, tuples, and arrays, especially when dealing with          large numerical data. 
        (1) Memoryvies are typed, meaning cython knows the data type.
        (2) Memoryviews can disable bounds checking and wraparound behavior for additional performance gains.
        (3) Data is stored contiguously on memory
        (4) They avoid the need of copying data to c functions.

--> Python: tuple

    Cython Type: No direct equivalent, usually managed as Python objects
        If the stored data types are same, it can be converted to np.ndarray object and and can be stored in memoryviews.

--> Python dict

    Cython Type: No direct equivalent, managed as Python objects
    Conversion: Use cdef dict to declare dictionaries

--> Python set

    Cython Type: No direct equivalent, managed as Python objects
    Conversion: Use cdef set to declare sets


## Explanations of Commands Used
---
### -Setup- files

Here is a sample from this repository with added explanations using '#'. (./SpeedTesting/setup_cuda.py):
    
    #'Extension': This is used to define extension modules that need to be compiled.
    #'get_python_inc': This function returns the path to the Python include directory, necessary for compiling extensions.
    #'setup': This is the main function from setuptools used to set up the Python package.
    
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
    from distutils.sysconfig import get_python_inc
    import numpy as np
    
    #'custom_build_ext' class is used to customize the build process, enabling the compilation of CUDA code within the setup script.
    class custom_build_ext(build_ext): 
        #Overriden method. To modify the extension building process.
        def build_extensions(self):
            self.compiler.src_extensions.append('.cu') #Adds .cu to the list of source file extensions that the compiler will recognize.
            original_compile = self.compiler._compile #'self.compiler._compile' is the original function used by the compiler to compile source files.

            #Defines a new compile method that distinguishes between CUDA (.cu) and other source files.
            def new_compile(obj, src, ext, cc_args, extra_postargs, pp_opts): 
                if src.endswith('.cu'):
                    nvcc_args = extra_postargs.get('nvcc', [])
                    self.spawn(['nvcc', '-c', src,
                                '-o', obj,
                                '-arch=sm_75', '-Xcompiler', '-fPIC'] + nvcc_args) #'-arch=sm_75': Target architecture (specific to the GPU).
                else:
                    gcc_args = extra_postargs.get('gcc', [])
                    original_compile(obj, src, ext, cc_args, gcc_args, pp_opts)
            
            self.compiler._compile = new_compile #Replaces the compiler's compile method with the new one.
            build_ext.build_extensions(self) #Calls the original build_extensions method to build the extensions.
    
    #Defines a list of extension modules to be compiled.
    #'sources': Lists the source files for the extension. This includes a Cython file (cythonCu.pyx) and a CUDA file (matrixSum.cu).
    #'include_dirs': Specifies directories to search for header files. Includes numpy headers and Python headers.
    #'library_dirs': Specifies directories to search for libraries.
    #'libraries': Lists libraries to link against. Here, it links against the CUDA runtime library (cudart).
    #'language': Specifies the language for the extension module.
    #'runtime_library_dirs': Specifies directories to search for runtime libraries.
    ext_modules = [
        Extension(
            "wrappedCuda",
            sources=["./Cython/cythonCu.pyx", "./Codes/matrixSum.cu"], 
            include_dirs=[np.get_include(), get_python_inc()],
            library_dirs=["/usr/local/cuda/lib64"], 
            libraries=["cudart"],
            language="c++", 
            extra_compile_args={'gcc': ["-std=c++11", "-fPIC"],
                                'nvcc': ['-arch=sm_75']},
            runtime_library_dirs=["/usr/local/cuda/lib64"] 
        )
    ]
    
    setup(
        name="speedTest",
        ext_modules=ext_modules, #Specifies the extension modules to be compiled.
        cmdclass={'build_ext': custom_build_ext}, #Uses the custom build class to handle the build process.
        zip_safe=False,
    )
