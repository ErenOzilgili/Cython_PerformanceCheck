# Person -Overview-
---
  This is a folder for wrapped version of the (person.cpp) file. In it, we have struct and class structures. They are used from python with the help of the
  cython.

# Sample Run
---
  After dependencies are satisfied, cd into the folder (Person).

    cd ./Person
  
  Afterwards, running the following commands will help you run the sample code (testPerson.py).

    python setup.py build_ext --inplace

    python testPerson.py 

  Below is a sample run:

    (C++) Age updated to 21
    (Cython) You are 21 now
