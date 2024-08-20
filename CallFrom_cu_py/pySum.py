import timeit #Used to measure the time of the function executions
import wrappedCuda #Cuda

"""
number = 100

total = timeit.timeit(stmt="wrappedCuda.cyFuncX()",
                        setup="import wrappedCuda",
						number=100)

print("Repeating the function {} time, below are the execution times".format(number))
print("Time it took for -cuda wrapped with cython- is (sec): {}\n".format(total))
"""

wrappedCuda.cyFuncX()

