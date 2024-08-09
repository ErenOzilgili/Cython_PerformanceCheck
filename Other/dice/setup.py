from setuptools import setup,Extension

from Cython.Build import cythonize

extension = Extension(
    name="dice",
    sources=["dc.pyx","dice.cpp","randgen.cpp"],
    language="c++"
)


setup(name="dice",
    ext_modules=cythonize(extension))