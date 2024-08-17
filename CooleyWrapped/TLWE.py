import numpy as np
#import cython function
from tfhe_fft import performOp_Cy, mulPolys

"""
    General Notes:
        Define as trivial the samples having the mask a = 0 and
        noiseless the samples having the standard deviation alpha = 0.
        
"""

#Define a data type for usage in float2 of cuda.
float2_dtype = np.dtype([('x', np.float32), ('y', np.float32)])

class ParametersTLWE:
    def __init__(self, k:int, N:int, alpha:float) :
        """
            Constructor for the parameters used to sample data.

            alpha --> Represents the standart deviation in float.
                k --> k is the vector dimension of vector a.
                      (k+1) also represents the dimension of the TLWE ciphertext.
        """
        self.k = k
        self.N = N
        self.alpha = alpha
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
        self.s = [(list)( np.rint(np.random.uniform(0, 1, size=self.N))) for i in range(0, self.k)]

    def sample_a(self):
        """
            Uniformly sample a.

            a is a vector of length k consisting of
            uniformly sampled polynomials in TN[X]^k.
            Produced per message and returned.
        """
        return [(list)( np.random.uniform(0, 1, size=self.N) ) for i in range(0, self.k)]

    def sample_error(self):
        """
            Sample error from TN[X] with Gaussian distribution with standart deviation alpha.

            Produced per message and returned.
        """
        mean = 0 #mean is the expected average of the distribution. It can vary depending on the choice.
        error = np.random.normal(loc=mean, scale=self.alpha, size=self.N)

        return np.mod(error, 1)

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
        #Threshold for deciding between python and cuda.
        if True:#Cuda
            #Note: Consider using double2 in cuda, maybe.
            #Create an array of float2

            """
                Below two lines should be deleted, only tryouts.
            """
            x = [2, 3, 4, 1]
            y = [1, 2, 1, 1]
            prepRes = [0, 0, 0, 0, 0, 0, 0, 0]
            arrX = np.array([(val, 0.0) for val in x], dtype=float2_dtype)

            arrY = np.array([(val, 0.0) for val in y], dtype=float2_dtype)

            prep = np.array([(val, 0.0) for val in prepRes], dtype=float2_dtype)

            """
            #fft and invfft
            coeffToGetX = performOp_Cy(arrX, self.N)

            print("Print the list X after FFT and invFFT:")
            print(coeffToGetX)

            #print("Print the list X:")
            #print(arrY)
            """
            arrXPadded = np.pad(arrX, (0, 4), mode='constant', constant_values=(0.0, 0.0))
            arrYPadded = np.pad(arrY, (0, 4), mode='constant', constant_values=(0.0, 0.0))
            #resultOfPolyMul = np.pad(z, (0, 2 * 4), mode='constant', constant_values=(0.0, 0.0))

            resultOfPolyMul = mulPolys(arrXPadded, arrYPadded, prep, 8)

            print("Result:")
            for i in range(0, 8):
                print(i, "th result: ", np.round(resultOfPolyMul[i][0]))
            

        else:#Python
            pass

        return None #TODO change later on
        
    def dotProductPolys(self, x:list, y:list):
        """
            a
        """
        #Initialize the list that will hold the ultimate result with zeros.
        sumIndDots = [0] * self.N

        for i in range(0, self.k):
            sumIndDots = sum(i) in zip(sumIndDots, self.mulPolys(x[i], y[i])) #Each of x[i] and y[i] have N coefficients.
            
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
sample = ParametersTLWE(k = 4, N = 8, alpha=0)
print("Secret key is sampled like this:")
print(sample.s[0])
print("FFT and invFFT on secret key:")
sample.mulPolys(sample.s[0], sample.s[0]) #Pass only shape(N) list. There are k of them in sample.s
