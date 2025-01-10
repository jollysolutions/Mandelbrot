from setuptools import setup, Extension
import site

module1 = Extension('gmp_mandel',
                    sources = ['pure_c.c'],
                    libraries = ['gmp'],
                    include_dirs = [site.getsitepackages(), "D:\gmp\gmp-6.3.0", "D:\gmp\gmpy\src", "D:\gmp\mpfr\src", "D:\gmp\mpc\src", "DC:\msys64\\ucrt64\lib", "C:\\Users\jolly\\anaconda3\Lib\site-packages\gmpy2.libs"] )

setup (name = 'gmp_mandel',
       version = '1.0',
       description = 'A Mandelbrot function in C using GMP',
       ext_modules = [module1],
       requires = ['gmpy2'] )

# python setup.py build_ext --inplace --compiler=mingw32
