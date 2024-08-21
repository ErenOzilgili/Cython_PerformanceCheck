import timeit #Used to measure the time of the function executions
import wrappedCuda #Cuda


number = 100

prep="""
import wrappedCuda
wrappedCuda.resetDev()
"""

total = timeit.timeit(stmt="wrappedCuda.cyFuncX()",
                       setup=prep,
					    number=100)

print("Repeating the function {} time, below are the execution times".format(number))
print("Time it took for -cuda wrapped with cython- is (sec): {}".format(total))
print("Time it took for -cuda wrapped with cython- is (sec) (amortized): {}\n".format(total/number))



