from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "wrappedCyC",
        sources=["./Cython/cython.pyx", "./Codes/matrixSum.cpp"],
        language='c++',
        include_dirs=[np.get_include()]
    )
]

setup(
    name="speedTest",
    ext_modules=cythonize(ext_modules)
)