# Cython_PerformanceCheck
A library with examples to show the performance gains due to usage of cython. It will be used to demonstrate how and why efficiency increases when used cython.

## Dependencies

## Cython Speed
--> First of all, python is an interpreted language, which means that python code won't be directly compiled into machine code. On the other hand, cython     will compile the python into C code, then it will be compiled into machine code. Resulting machine code will run faster than the interpreted code.

--> Python is dynamically typed, which means that variable types are determined and checked at runtime rather than during compilation. On the other hand,     Cython allows for static typing, meaning you can declare the types of variables explicitly. This feature of cython prevents the overhead associated       with dynamic typing in Python, leading to faster execution.

--> Since Cython code is closer to C, it has less overhead than Python. This includes reduced function call overhead, memory management optimizations,        and more efficient looping constructs.

--> Cython can use C-level memory management, which can be more efficient than Python’s garbage collector, especially for large datasets or performance-     critical applications.

--> Cython seamlessly integrates with C and C++ libraries, allowing for the use of highly optimized external libraries directly in the code, which can       enhance performance.
    
## Data Conversions



## Explanations of Commands
### Setup.py Files
