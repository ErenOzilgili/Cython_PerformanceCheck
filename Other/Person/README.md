# Person -Overview-
---
  This is a folder for wrapped version of the (person.cpp) file. In it, we have struct and class structures. They are used from python with the help of the
  cython.

## Dependencies
---
  Apart from python, cython3 and setuptools can be installed via 
  
    pip install setuptools 
  
    pip install cython3

## Sample Run
---
  After dependencies are satisfied, cd into the folder (Person).

    cd ./Person
  
  Afterwards, running the following commands will help you run the sample code (testPerson.py).

    python setup.py build_ext --inplace

    python testPerson.py 

  Below is a sample run:

    Updating age... with int = 21
    (C++) Age updated to 21
    (Cython) You are 21 now
    Couldn't catch the error type
    Name: Eren
    Age: 21
    Address: Ulu, Istanbul, Marmara 210
    
    ------------------
    Error thrown from cython itself:
    Updating age... with string = "a"
    Error has been encountered! String is passed when expected int
    
    --------------------
    Error thrown from c++ and caught in cython:
    RuntimeError --- Printed in try-except block except part
    
    --------------------
    Error thrown from c++ and caught in python:
    Catched exception thrown from c++ in python

