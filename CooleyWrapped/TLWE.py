import numpy as np
#import cython function
from tfhe_fft import performOp_Cy, mulPolys, schoolBookMul
import timeit

"""
    General Notes:
        Define as trivial the samples having the mask a = 0 and
        noiseless the samples having the standard deviation alpha = 0.
        
"""

#Define a data type for usage in float2 of cuda.
float2_dtype = np.dtype([('x', np.float32), ('y', np.float32)])

class ParametersTLWE:
    def __init__(self, k:int, N:int, sigma:float, q:int, p:int) :
        """
            Constructor for the parameters used to sample data.

            sigma --> Represents the standart deviation in float.
                k --> k is the vector dimension of vector a.
                      (k+1) also represents the dimension of the TLWE ciphertext.
        """
        self.k = k
        self.N = N
        self.sigma = sigma
        self.q = q
        self.p = p
        self.sample_s() #Sample the secret key

    def sample_s(self):
        """
            For uniformly sampling the secret key.

            s is a list that is an element of BN[X]^k.
            This means that secret key s (SK)
            will be k entries in length list with 
            each of them consisting of a list of length N
            uniformly sampled coefficients in {0, 1}.
        """
        #TODO change this
        self.s = [(list)( np.rint(np.random.uniform(0, 1, size=self.N)) ) for i in range(0, self.k)]

    def sample_a(self):
        """
            Uniformly sample a.

            a is a vector of length k consisting of
            uniformly sampled polynomials in TN[X]^k.

            Although we are sampling x = mod q, remember that we
            are after all we are working in x/q which is an element of torus.
        """
        #return [(list)( np.random.uniform(0, 1, size=self.N) ) for i in range(0, self.k)]
        return  [(list)( np.random.randint(0, self.q, size=self.N) ) for i in range(0, self.k)]

    def sample_error(self):
        """
            Sample error from TN[X] with Gaussian distribution with standart deviation alpha.

            Produced per message and returned.
        """
        mean = 0 #mean is the expected average of the distribution. It can vary depending on the choice.
        error = np.random.normal(loc=mean, scale=self.sigma, size=self.N)
        #print(error)

        #TODO take a look at how the rounding is being made SoK page 7.
        return (list)(np.round(self.q * error))

    def encrypt(self):
        """
            Produce the ciphertext of the form (a,b)

            (a,b) is, in total, dimension of (k+1).
            a itself is of dimension of k in TN[X]^k.
            b is of dimension 1 and is produced by a.s + scale.message + error ??????????
            where scale denotes the scaling factor.
        """
        pass

    def encode(self, message, error):
        pass

    def func_phase(self, ciphertext:list):
        """
            Performs part of the decryption process.

            Use the ciphertext (a,b) of length (k+1)
            to decrypt using the function called phase.
            Phase is descryped as ( f = b - a.s ) where 
            a.s denotes the dot product of the two vectors.
            Note that s is SK (secret key).
        """
        return self.subtractPolys(ciphertext[1], self.dotProductPolys(self.s, ciphertext[0]))

    def mulPolys(self, x:list, y:list):
        """
            Mutiply polinomials given with their coeffcients as list.

            Take the dot product of vector a
            and secret key SK. Use cuda and perform fft.
        """

        """
        #Threshold for deciding between python and cuda.
        if True:#Cuda
            #Note: Consider using double2 in cuda, maybe.
            #Create an array of float2
            x = (2**7) * [2, 3, 4, 1] #--> size of 2**9
            y = (2**7) * [1, 2, 1, 1]
            prepRes = (2**7) * [0, 0, 0, 0, 0, 0, 0, 0] #--> Result will be of size 2**10

            arrX = np.array([(val, 0.0) for val in x], dtype=float2_dtype)

            arrY = np.array([(val, 0.0) for val in y], dtype=float2_dtype)

            prep = np.array([(val, 0.0) for val in prepRes], dtype=float2_dtype)
            
            arrXPadded = np.pad(arrX, (0, 2**9), mode='constant', constant_values=(0.0, 0.0))#Pad with 2**9 elements to reach 2**10 elements
            arrYPadded = np.pad(arrY, (0, 2**9), mode='constant', constant_values=(0.0, 0.0))
            #resultOfPolyMul = np.pad(z, (0, 2 * 4), mode='constant', constant_values=(0.0, 0.0))

            resultOfPolyMul = mulPolys(arrXPadded, arrYPadded, prep, 2**10)

            print("Result:")
            for i in range(0, 2**10):
                print(i, "th result: ", np.round(resultOfPolyMul[i][0]))
            

        else:#Python
            pass
        """
        assert len(x) == len(y)

        lenMul = len(x)
        prepRes = (2 * lenMul) * [0]

        #Convert for type compatibility with cuda's float2
        resultOfPolyMul = np.array([(val, 0.0) for val in prepRes], dtype=float2_dtype)
        float2X = np.array([(val, 0.0) for val in x], dtype=float2_dtype)
        float2Y = np.array([(val, 0.0) for val in y], dtype=float2_dtype)

        #Pad as necessary
        float2XPadded = np.pad(float2X, (0, lenMul), mode='constant', constant_values=(0.0, 0.0))
        float2YPadded = np.pad(float2Y, (0, lenMul), mode='constant', constant_values=(0.0, 0.0))

        #Result of the polynomial multiplication 
        resultOfPolyMul = mulPolys(float2XPadded, float2YPadded, resultOfPolyMul, 2*lenMul)
        
        #Take the integer parts of the float2_dtype
        polyMul = [np.round(val['x']) for val in resultOfPolyMul]

        #TODO may not be (list). Think.
        return (list)(polyMul)
        
    def dotProductPolys(self, x:list, y:list):
        """
            x may represent the vector-a
            y may represent the vector-s

            vector-a . vector-s
        """
        #Result will be stored in this list.
        sumIndDots = 2 * self.N * [0]

        for i in range(0, self.k):
            #Call cuda using mulPolys and sum the individual polynomials coefficient wise.
            sumIndDots = [(a + s) for a, s in zip(sumIndDots, self.mulPolys(x[i], y[i]))] #Each of x[i] and y[i] have N coefficients.
            
        return sumIndDots 
    
    def addPolys(self, x:list, y:list):
        """
            Add the coefficients off the two polynomials represented as two lists.
        """
        #Set a threshold according to N
        if True:
            return [x[i]-y[j] for i,j in range(0, self.N)] 
        else:
            pass

    def subtractPolys(self, x:list, y:list):
        """
            Subtract two polynomials given with their coefficients given as a list.
        """
        #Set a threshold for N
        if True:         
            return [x[i]-y[j] for i,j in range(0, self.N)] 
        else:
            pass

#Testing
#--------------
sigma=2**(-25)
sample = ParametersTLWE(k = 1, N = 2**10, sigma=sigma, q=8, p=4)

print("Secret key is sampled like this:")
print(sample.s)

print("Vector-a is sampled like this: ")
a = sample.sample_a()
print(a)

print("Error: ")
error = sample.sample_error()
print(error)

print("(b - error) is (vector-a . vector-s): ")
#TODO
#Remember that here we are printing 2*N coeffs, but it should have (2 * N) - 1
#Later reduce this mod X^N + 1
print(sample.dotProductPolys(sample.s, a))

"""
sumIndDots = [(0, 0), (0,0)]
sumIndDots = [(a[0] + s[0], a[1] + s[1]) for a, s in zip(sumIndDots, [(1, 2), (3, -4)])] #Each of x[i] and y[i] have N coefficients.

print(sumIndDots)
"""














"""
def callCudaForSpeed():
    resultOfPolyMul = mulPolys(arrXPadded, arrYPadded, prep, 2**10)
    return resultOfPolyMul
def callCythonForSpeedSchool():
    resultOfPolyMul = schoolBookMul(arrX, arrY, prep, 2**9)
    return resultOfPolyMul

def mulSpeedTest():
        if True:

            x = (2**7) * [2, 3, 4, 1] #--> size of 2**9
            y = (2**7) * [1, 2, 1, 1]
            prepRes = (2**7) * [0, 0, 0, 0, 0, 0, 0, 0] #--> Result will be of size 2**10

            global arrX
            arrX = np.array([(val, 0.0) for val in x], dtype=float2_dtype)
            global arrY
            arrY = np.array([(val, 0.0) for val in y], dtype=float2_dtype)

            x = (2**7) * [2, 6, 1, -1] #--> size of 2**9
            y = (2**7) * [2, 4, 3, 0]
            global arrXPadded
            arrXPadded = np.pad(arrX, (0, 2**9), mode='constant', constant_values=(0.0, 0.0))#Pad with 2**9 elements to reach 2**10 elements 
            global arrYPadded
            arrYPadded = np.pad(arrY, (0, 2**9), mode='constant', constant_values=(0.0, 0.0))

            #Shared
            global prep
            prep = np.array([(val, 0.0) for val in prepRes], dtype=float2_dtype)
            #resultOfPolyMul = np.pad(z, (0, 2 * 4), mode='constant', constant_values=(0.0, 0.0))
            timeOfFFTCuda = timeit.timeit(stmt="callCudaForSpeed()", setup="from __main__ import callCudaForSpeed", number=1)
            resultOfPolyMulCu = callCudaForSpeed()

            print("Result:")
            for i in range(0, 2**10):
                print(i, "th result: ", np.round(resultOfPolyMulCu[i][0]))

            timeOfFFTCySchool = timeit.timeit(stmt="callCythonForSpeedSchool()", setup="from __main__ import callCythonForSpeedSchool", number=1)
            resultOfPolyCy = callCythonForSpeedSchool()

            print("Result:")
            for i in range(0, 2**10):
                print(i, "th result: ", np.round(resultOfPolyCy[i][0]))

            for i in range (0, 2**10):
                assert np.round(resultOfPolyCy[i][0]) == np.round(resultOfPolyMulCu[i][0])
            print("Succesful")

            print("Cuda: ", timeOfFFTCuda)
            print("Cython: ", timeOfFFTCySchool)

        else:#Python
            pass

        return None #TODO change later on

mulSpeedTest()
"""
