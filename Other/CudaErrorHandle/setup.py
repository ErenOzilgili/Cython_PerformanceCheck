from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np

class custom_build_ext(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        original_compile = self.compiler._compile
        
        def new_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith('.cu'):
                nvcc_args = extra_postargs.get('nvcc', [])
                self.spawn(['nvcc', '-c', src,
                            '-o', obj,
                            '-arch=sm_75', '-Xcompiler', '-fPIC'] + nvcc_args)
            else:
                gcc_args = extra_postargs.get('gcc', [])
                original_compile(obj, src, ext, cc_args, gcc_args, pp_opts)
        
        self.compiler._compile = new_compile
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        "wrapper",
        sources=["wrapper.pyx", "ErrorSpecifier.cu"],
        include_dirs=[np.get_include(), "/usr/local/cuda/include"],
        library_dirs=["/usr/local/cuda/lib64"],
        libraries=["cudart"],
        language="c++",
        extra_compile_args={'gcc': ["-std=c++11", "-fPIC"],
                            'nvcc': ['-arch=sm_75']},
        runtime_library_dirs=["/usr/local/cuda/lib64"]
    )
]

setup(
    name="wrapper",
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': custom_build_ext},
    zip_safe=False,
)