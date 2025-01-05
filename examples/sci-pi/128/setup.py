from setuptools import setup, Extension
import numpy as np

module1 = Extension('int128_mandel',
                    sources = ['pure_c128.c'],
                    include_dirs = [np.get_include()] )

setup (name = 'int128_mandel',
       version = '1.0',
       description = 'A Mandelbrot function in C using 128 bit ints',
       ext_modules = [module1] )
