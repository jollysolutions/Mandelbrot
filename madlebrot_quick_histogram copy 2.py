from matplotlib import pyplot as plt
from matplotlib import colors
import timeit
import numpy as np
from numba import jit, vectorize, guvectorize, float64, complex64, int32, float32, int64, float64, complex128, cuda
from functools import cache
from time import time
from pprint import pp

@guvectorize([(int64[:], int64[:], int64[:], int64[:], int64[:])], '(n),(m),(),()->(n)',target='parallel')
def numpy_histogram_colours3(arr_in, histogram, sum, maxint, arr_out):
    # Histogram Colours Start
    arr_float = np.zeros_like(arr_in)
    histogram_sum = np.cumsum(histogram)
    for i in range(arr_in.shape[0]):
        arr_float[i] = float(histogram_sum[arr_in[i]]/sum[0]*maxint[0])
    arr_out[:] = arr_float[:]
    # Histogram Colours End

@cache
def mandelbrot_image3(xmin,xmax,ymin,ymax,width=3,height=3,maxiter=80,cmap='hot'):
    dpi = 72
    img_width = dpi * width
    img_height = dpi * height
    x,y,z, hist_sum, hist_total = mandelbrot_set3(xmin,xmax,ymin,ymax,img_width,img_height,maxiter)
    z_h = numpy_histogram_colours3(z, hist_sum, hist_total, maxiter)
    fig, ax = plt.subplots(figsize=(width, height),dpi=72)
    ticks = np.arange(0,img_width,img_width/10)
    x_ticks = xmin + (xmax-xmin)*ticks/img_width
    plt.xticks(ticks, x_ticks)
    y_ticks = ymin + (ymax-ymin)*ticks/img_height
    plt.yticks(ticks, y_ticks)

    norm = colors.PowerNorm(0.3)
    ax.imshow(z_h.T,cmap=cmap,origin='lower',norm=norm,aspect='equal')
    plt.show()

@jit(int64(complex128, int64),nopython=True, cache=True)
def mandelbrot3(c,maxiter):
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
            return 0
        period += 1
        if period > 20:
            xold = x
            yold = y
    return 0

@guvectorize([(complex128[:], int64[:], int64[:], int64[:], int64[:], int64[:])], '(n),(m),()->(n),(m),()',target='parallel')
def mandelbrot_numpy3(c, h, maxit, output, hist_t, sum_t):
    maxiter = maxit[0]
    hist = [0] * maxiter
    sum = 0
    for i in range(c.shape[0]):
        out = mandelbrot3(c[i],maxiter)
        output[i] = out
        hist[out] = hist[out] + 1
        sum = sum + out
    sum_t[:] = sum
    hist_t[:] = hist[:]

@cache
def mandelbrot_set3(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.longdouble)
    r2 = np.linspace(ymin, ymax, height, dtype=np.longdouble)
    c = r1 + r2[:,None]*1j
    h = np.linspace(0,0,maxiter,dtype=np.int64)
    n3, hist, sum = mandelbrot_numpy3(c, h, maxiter)
    hist_sum = np.array(maxiter)
    hist_sum = np.sum(hist.T,axis=1)
    hist_total = sum.sum()
    return (r1, r2, n3.T, hist_sum, hist_total)

@cache
def timing(xmin,xmax,ymin,ymax,width,height,maxiter):
    print("xmin : {}, xmax : {}, ymin : {}, ymax : {}, width : {}, height : {}, maxiter : {}".format(xmin,xmax,ymin,ymax,width,height,maxiter))
    scale = 72
    width = width * scale
    height = height * scale
    t = m = time()

    _, _ , arr_in, hist_sum, hist_total = mandelbrot_set3(xmin,xmax,ymin,ymax,width,height,maxiter)
    print("  mandelbrot : ", time()-m)
    h2 = time()
    arr_out = numpy_histogram_colours3(arr_in, hist_sum, hist_total, maxiter)
    print("  histogram2 : ", time()-h2)
    print("total : ", time()-t)
    print("----")

tt = time()
#for it in [1024, 2048, 4096, 8192]:
for it in [1024, 8192]:
    t_it = time()
    #for size in [100, 200, 400]:
    for size in [100]:
        timing(-2.0,0.5,-1.25,1.25,size,size,it)
        #timing(-0.74877,-0.74872,0.06505,0.06510,size,size,it)
        a = 0
    print("it time : ", time ()-t_it)
    print("----")
print("TOTAL : ", time()-tt)
mandelbrot_image3(-2.0,0.5,-1.25,1.25,width=100,height=100,maxiter=1024,cmap='gnuplot2')
mandelbrot_image3(-2.0,0.5,-1.25,1.25,width=100,height=100,maxiter=8192,cmap='gnuplot2')
#mandelbrot_image3(-0.74877,-0.74872,0.06505,0.06510,width=100,height=100,maxiter=1024,cmap='gnuplot2')
#mandelbrot_image3(-0.74877,-0.74872,0.06505,0.06510,width=100,height=100,maxiter=8192,cmap='gnuplot2')
#input("Press the Enter key to continue: ")
