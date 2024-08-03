from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        'wrappedPerson',
        sources=['wrapper.pyx', 'person.cpp'],
        language='c++',
    )
]

setup(
    name='classExample',
    ext_modules=cythonize(ext_modules)
)