import numpy as np
#import cython function
from tfhe_fft import mulPolys, schoolBookMul
import timeit

"""
    General Notes:
        Define as trivial the samples having the mask a = 0 and
        noiseless the samples having the standard deviation sigma = 0.
        
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
        self.sample_a()
        self.sample_s() #Sample the secret key
        self.publicKey(self.a, self.s)

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
        self.s = [list( np.rint(np.random.uniform(0, 1, size=self.N)) ) for i in range(0, self.k)]
        return self.s

    def sample_a(self):
        """
            Uniformly sample a.

            a is a vector of length k consisting of
            uniformly sampled polynomials in TN[X]^k.

            Although we are sampling x = mod q, remember that we
            are after all we are working in x/q which is an element of torus.
        """
        #return [list( np.random.uniform(0, 1, size=self.N) ) for i in range(0, self.k)]
        self.a = [list( np.random.randint(0, self.q, size=self.N) ) for i in range(0, self.k)]
        return self.a

    def sample_error(self):
        """
            Sample error from TN[X] with Gaussian distribution with standart deviation alpha.

            Produced per message and returned.
        """
        mean = 0 #mean is the expected average of the distribution. It can vary depending on the choice.
        error = np.random.normal(loc=mean, scale=self.sigma, size=self.N)
        #print(error)

        #TODO take a look at how the rounding is being made SoK page 7.
        self.error = list(np.round(self.q * error))
        return self.error
    
    def publicKey(self, a:list, s:list):
        #TODO look if k > 1. May work nevertheless.

        product = self.dotProductPolys(self.a, self.s)
        # Reduce the below mod X^N + 1
        productReduced = [(x - y) for x, y in zip(product[:self.N], product[self.N:2*self.N])]
        """
        print("ProductReduced: ")
        print(productReduced)
        """

        self.pk = [x for x in a] + [self.addPolys(productReduced, self.sample_error())]

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

    def func_phase(self):
        """
            Performs part of the decryption process.

            Use the ciphertext (a,b) of length (k+1)
            to decrypt using the function called phase.
            Phase is descryped as ( f = b - a.s ) where 
            a.s denotes the dot product of the two vectors.
            Note that s is SK (secret key).
        """
        #TODO look if k > 1. May wokr nevertheless.
        return self.subtractPolys(self.ciphertext[1], self.dotProductPolys(self.s, self.ciphertext[0]))

    def mulPolys(self, x:list, y:list):
        """
            Mutiply polinomials given with their coeffcients as list.

            Take the dot product of vector a
            and secret key SK. Use cuda and perform fft.
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
        return list(polyMul)
        
    def dotProductPolys(self, x:list, y:list):
        """
            x may represent the vector_a
            y may represent the vector_s

            (vector_a) . (vector_s)
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
            return [x[i] + y[i] for i in range(0, self.N)] 
        else:
            pass

    def subtractPolys(self, x:list, y:list):
        """
            Subtract two polynomials given with their coefficients given as a list.
        """
        #Set a threshold for N
        if True:         
            return [x[i]-y[i] for i in range(0, self.N)] 
        else:
            pass

#Testing
#--------------
sigma=2**(-2) #2**-25
N = 2**2
q = 2**4
p = 2**2
k = 2
sample = ParametersTLWE(k = k, N = N, sigma=sigma, q=q, p=p)

print("\nSecret key is sampled like this: ")
print(sample.s)

print("\nVector_a is sampled like this: ")
print(sample.a)

"""
print("\nvector_a . vector_s: ")
print(sample.dotProductPolys(sample.a, sample.s))
"""

print("\nPublic key: ")
print(sample.pk)

print("\nError was the below when public key were created: ")
print(sample.error)

"""
print("Secret key is sampled like this:")
print(sample.s)

print("vector_a is sampled like this: ")
a = sample.sample_a()
print(a)

print("Error: ")
error = sample.sample_error()
print(error)

print("(b - error) is  vector_a . vector_s): ")
#TODO
#Remember that here we are printing 2*N coeffs, but it should have (2 * N) - 1
#Later reduce this mod X^N + 1

#Reducing in a way
prod_as = sample.dotProductPolys(sample.s, a)
prod_as_reduced = [(x - y) for x, y in zip(prod_as[:N], prod_as[N:2*N])]

print(sample.dotProductPolys(sample.s, a))

print("Reduced --> (b - error) is  vector_a . vector_s): ")
print(prod_as_reduced)
"""
