# importing the required module
import timeit

# code snippets to be executed only once
mysetup1 = "import dice as dc"
mysetup2 = "import random as rd"
# code snippets whose execution times are to be measured(randomizers)
mycode1 = ''' 
def example(): 
    dicey = dc.pyDice(6)
    for x in range(100000): 
        dicey.Roll()
'''

mycode2 = '''
def example():
    for x in range(100000): 
        rd.randint(1,6)
    
    '''
# timeit statement(multiple repeats to find a stable time)




times2 = timeit.repeat(setup=mysetup2,
                    stmt=mycode2,
                    repeat=1,
                    number=10000000000000000)

# The smallest of them will be printed.



times1 = timeit.repeat(setup=mysetup1,
                    stmt=mycode1,
                    repeat=1,
                    number=10000000000000000)

print('The time of wrapped dice class method Roll() which utilizes a randomizer: {}'.format(min(times1)))
print('The time of pure Python module random\'s randint() method which utilizes a randomizer: {}'.format(min(times2)))
print('Wrapped dice class\'s Roll() method is faster than pure Python randint method by a fraction of: {}'.format((min(times1)/min(times2)*100)))