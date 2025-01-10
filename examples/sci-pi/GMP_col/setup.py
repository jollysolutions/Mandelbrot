from setuptools import setup, Extension
import numpy

module1 = Extension('gmp_mandel',
                    sources = ['pure_c2.c'],
                    libraries = ['gmp'],
                    include_dirs = [numpy.get_include()] )

setup (name = 'gmp_mandel',
       version = '1.0',
       description = 'A Mandelbrot function in C using GMP',
       ext_modules = [module1] )

# python setup.py build_ext --inplace --compiler=mingw32
