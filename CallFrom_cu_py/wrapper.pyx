cdef extern from "matrixSum.h":
    void funcX()
    void resetDevice()

def cyFuncX():
    funcX()

def resetDev():
    resetDevice()