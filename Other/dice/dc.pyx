# distutils: language = c++



cdef extern from "randgen.h":
    cdef cppclass RandGen:
        RandGen() except +
        int RandInt(int low,int max) except +


cdef extern from "dice.h":
    cdef cppclass Dice:
        Dice()  except +
        Dice(int) except +
        int Roll()
        int NumSides()
        int NumRolls()
        int myRollCount,mySides


cdef class pyDice:
    cdef Dice *c_dice
    
    
    def __cinit__(self,int mySides=5):
        self.c_dice = new Dice(mySides)
        self.c_dice.myRollCount = 0
    
    def __dealloc__(self):
        del self.c_dice
    
    def Roll(self):
        cdef RandGen c_rand 
        self.c_dice.myRollCount += 1
        return c_rand.RandInt(1,self.c_dice.mySides) 
    
    def NumSides(self):
        return self.c_dice.mySides
    
    def NumRolls(self):
        return self.c_dice.myRollCount
    
    def DualRoll(self,pyDice dice):
        cdef int result = 0
        result += self.c_dice.Roll() + dice.c_dice.Roll()
        return result



