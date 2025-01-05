from setuptools import setup, Extension
import numpy as np

module1 = Extension('int256_mandel',
                    sources = ['pure_c256.c'],
                    include_dirs = [np.get_include()] )

setup (name = 'int256_mandel',
       version = '1.0',
       description = 'A Mandelbrot function in C using 128 bit ints and 256 bit products',
       ext_modules = [module1] )
