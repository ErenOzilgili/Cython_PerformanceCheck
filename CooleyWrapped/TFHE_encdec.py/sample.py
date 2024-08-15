import numpy as np

class ParametersTLWE:
    def __init__(self, k:int, N:int, alpha:float) :
        """Constructor for the parameters used to sample data.

            alpha --> Represents the standart deviation in float.
                k --> k is the vector dimension of vector a.
                      (k+1) also represents the dimension of the TLWE ciphertext.
        """
        self.k = k
        self.N = N
        self.alpha = alpha
        self.sample_s() #Sample the secret key

    def sample_s(self):
        """For uniformly sampling the secret key.

            s is a tuple that is an element of BN[X]^k.
            This means that secret key s (SK)
            will be k entries in length list with 
            each of them consisting of tuples of length N
            uniformly sampled coefficients in {0, 1}.
        """
        self.s = [(tuple)( np.rint(np.random.uniform(0, 1, size=self.N))) for i in range(0, self.k)]

    def sample_a(self):
        """Uniformly sample a.

            a is a vector of length k consisting of
            uniformly sampled polynomials in TN[X]^k.
            Produced per message and returned.
        """
        return [(list)( np.random.uniform(0, 1, size=self.N) ) for i in range(0, self.k)]

    def sample_error(self):
        """Sample error from TN[X] with Gaussian distribution with standart deviation alpha.

            Produced per message and returned.
        """
        mean = 0 #mean is the expected average of the distribution. It can vary depending on the choice.
        error = np.random.normal(loc=mean, scale=self.alpha, size=self.N)

        return np.mod(error, 1)


    def dotProduct(self, a:list):
        """Dot product for the vector multiplications.

            Take the dot product of vector a
            and secret key SK. Use cuda and perform fft.
        """
        pass

    def encrypt(self):
        """Produce the ciphertext of the form (a,b)

            (a,b) is, in total, dimension of (k+1).
            a itself is of dimension of k in TN[X]^k.
            b is of dimension 1 and is produced by a.s + scale.message + error ??????????
            where scale denotes the scaling factor.
        """
        pass

    def encode(self, message, error):
        pass

    def func_phase(self, ciphertetx:list):
        """Performs part of the decryption process.

            Use the ciphertext (a,b) of length (k+1)
            to decrypt using the function called phase.
            Phase is descryped as ( f = b - a.SK ) where 
            a.SK denotes the dot product of the two vectors.
        """
        pass


#Testing
#--------------
a = ParametersTLWE(k = 2, N = 4, alpha=0)
a.sample_s()
print(a.s)