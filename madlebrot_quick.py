from matplotlib import pyplot as plt
from matplotlib import colors
import timeit
import numpy as np
from numba import jit, vectorize, guvectorize, float64, complex64, int32, float32, int64, float64, complex128, cuda
from functools import cache

@cache
def mandelbrot_image(xmin,xmax,ymin,ymax,width=3,height=3,maxiter=80,cmap='hot'):
    dpi = 72
    img_width = dpi * width
    img_height = dpi * height
    x,y,z = mandelbrot_set(xmin,xmax,ymin,ymax,img_width,img_height,maxiter)
    
    fig, ax = plt.subplots(figsize=(width, height),dpi=72)
    ticks = np.arange(0,img_width,img_width/10)
    x_ticks = xmin + (xmax-xmin)*ticks/img_width
    plt.xticks(ticks, x_ticks)
    y_ticks = ymin + (ymax-ymin)*ticks/img_width
    plt.yticks(ticks, y_ticks)
    
    norm = colors.PowerNorm(0.3)
    ax.imshow(z.T,cmap=cmap,origin='lower',norm=norm,aspect='equal')
    plt.show()

@jit(int64(complex128, int64),nopython=True, cache=True)
def mandelbrot(c,maxiter):
    nreal = 0
    real = 0
    imag = 0
    for n in range(maxiter):
        nreal = real*real - imag*imag + c.real
        imag = 2* real*imag + c.imag
        real = nreal;
        if real * real + imag * imag > 4.0:
            return n
    return 0

@jit(int64(complex128, int64),nopython=True, cache=True)
def mandelbrot2(c,maxiter):
    x0 = np.longdouble(c.real)
    y0 = np.longdouble(c.imag)
    x025 = x0-0.25
    x1 = x0+1.0
    q = (y0*y0) + (x025*x025)
    if (q*(q+x025)) <= 0.25*y0*y0:
        return 0
    if ((x1*x1) + (y0*y0)) <= 0.0625:
        return 0
    x = np.longdouble(0.0)
    y = np.longdouble(0.0)
    x2 = np.longdouble(0.0)
    y2 = np.longdouble(0.0)
    xold = np.longdouble(0.0)
    yold = np.longdouble(0.0)
    period = 0
    for n in range(maxiter):
        y = 2*x*y + y0
        x = x2 - y2 + x0
        x2 = x*x
        y2 = y*y
        if x2 + y2 >= 4.0:
            return n
        if x == xold and y == yold:
            return maxiter
        period += 1
        if period > 20:
            xold = x
            yold = y
    return 0

@guvectorize([(complex128[:], int64[:], int64[:])], '(n),()->(n)',target='parallel')
def mandelbrot_numpy(c, maxit, output):
    maxiter = maxit[0]
    for i in range(c.shape[0]):
        output[i] = mandelbrot2(c[i],maxiter)

@cache
def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.longdouble)
    r2 = np.linspace(ymin, ymax, height, dtype=np.longdouble)
    c = r1 + r2[:,None]*1j
    n3 = mandelbrot_numpy(c,maxiter)
    return (r1,r2,n3.T)

it = 4096
print(timeit.timeit('mandelbrot_set(-2.0,0.5,-1.25,1.25,140*72,140*72,it)', number = 1, globals=globals()))

print(timeit.timeit('mandelbrot_set(-0.74877,-0.74872,0.06505,0.06510,140*72,140*72,it)', number = 1, globals=globals()))

mandelbrot_image(-2.0,0.5,-1.25,1.25,width=140,height=140,maxiter=it,cmap='gnuplot2')
mandelbrot_image(-0.74877,-0.74872,0.06505,0.06510,width=140,height=140,maxiter=it,cmap='gnuplot2')
