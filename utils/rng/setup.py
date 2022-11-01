from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("utils/rng/getEJrandnums.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)
