/* Calculate a column of the Mandelbrot Set, using fixed point arithmetic
 * with __int128 and interfacing with python / numpy.
 *
 * This version works with bit shifts of up to 122
 *
 * MJR 8/2022
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

/* Convert a PyLong to an __int128 */

static __int128 PyLong_AsInt128(PyLongObject *in){
  Py_ssize_t i,digits,sign,is_neg;
  __int128 out,tmp;

  if (!PyLong_Check(in)) return -1;
  
  sign=1;
  is_neg=-1;
  digits=in->long_value.lv_tag >> 3;
  if (digits==0) return 0;

  if (digits<0){
    sign=-1;
    digits=-digits;
  }
  is_neg=(in->long_value.lv_tag & 3) == 2;
  
  out=in->long_value.ob_digit[0];

  for(i=1;i<digits;i++){
    tmp=in->long_value.ob_digit[i];
    tmp=tmp<<(i*PyLong_SHIFT);
    out+=tmp;
  }

  if (sign==-1) out=-out;
  if (is_neg) out=-out;

  return out;
}

/* Multiply two __int128s with a shift */

static __int128 mul_i128(__int128 a, __int128 b, long bshift){
  __int128 out;
  uint64_t a_hi,a_lo,b_hi,b_lo;
  unsigned __int128 tmp;
  int sign;

  sign=(a^b)>>127;

  /* intmax_t is 64 bits, so cannot use imaxabs here */
  //  a=imaxabs(a);
  if (a>>127) a=-a;

  //  b=imaxabs(b);
  if (b>>127) b=-b;

  a_lo=(a&0xffffffffffffffff);
  b_lo=(b&0xffffffffffffffff);

  a_hi=(a>>64);
  b_hi=(b>>64);

  out=(a_lo*(unsigned __int128)b_lo)>>bshift;

  /* a_hi and b_hi both have zero as msb, so this will not overflow */
  tmp=(a_lo*(unsigned __int128)b_hi)+(b_lo*(unsigned __int128)a_hi);
  if (bshift>64) tmp=tmp>>(bshift-64);
  else if (bshift<64) tmp=tmp<<(64-bshift);
  out+=tmp;

  /* Should check for whether bshift>128,
     but it never will be for this usage */
  out+=(a_hi*(unsigned __int128)b_hi)<<(128-bshift);

  if (sign) out=-out;

  return out;
}

/* Square an __int128 with a shift */

inline unsigned __int128 sq_i128(__int128 a, long bshift){
  unsigned __int128 out,tmp;
  uint64_t a_hi,a_lo;

  /* intmax_t is 64 bits, so cannot use imaxabs here */
  //  a=imaxabs(a);
  if (a>>127) a=-a;

  a_lo=(a&0xffffffffffffffff);
  a_hi=(a>>64);

  out=(a_lo*(unsigned __int128)a_lo)>>bshift;

  /* Absorb a *2 into the shift */
  tmp=a_lo*(unsigned __int128)a_hi;
  if (bshift>65) tmp=tmp>>(bshift-65);
  else if (bshift<65) tmp=tmp<<(65-bshift);
  out+=tmp;

  /* Should check for whether bshift>128,
     but it never will be for this usage */
  out+=(a_hi*(unsigned __int128)a_hi)<<(128-bshift);

  return out;
}


static PyObject *mandel_column(PyObject *self, PyObject *args){
  int size,maxiter;
  long bshift;
  PyObject *x0_py,*ymin_py,*ymax_py,*col;
  __int128 x,y,x0,y0,ymin,ymax,ystep;
  unsigned __int128 x2,y2,mod_test;
  int i,j,y_remainder;
  npy_intp dims[1];
  
  if (!PyArg_ParseTuple(args,"OOOiil",&x0_py,&ymin_py,&ymax_py,
			&size,&maxiter,&bshift)){
    fprintf(stderr,"Argument error\n");
    exit(1);
  }
  
  dims[0]=size;
  col=PyArray_SimpleNew(1,dims,NPY_DOUBLE);
  
  x0=PyLong_AsInt128((PyLongObject*)x0_py);
  ymin=PyLong_AsInt128((PyLongObject*)ymin_py);
  ymax=PyLong_AsInt128((PyLongObject*)ymax_py);

  mod_test=4;
  mod_test<<=bshift;

  ystep=(ymax-ymin)/size;
  y_remainder=(ymax-ymin)%size;
  
  for(j=0;j<size;j++){

    y0=ymax-j*ystep-(j*y_remainder)/size;

    x=x0;
    y=y0;
    x2=sq_i128(x,bshift);
    y2=sq_i128(y,bshift);

    for(i=0;i<maxiter;i++){
      /* tmp=x2+y2 */
      if (x2+y2>=mod_test) break;
      /* y=((x*y)>>(bshift-1))+y0 */
      y=mul_i128(x,y,bshift-1)+y0;
      /* x=((x2-y2)>>bshift)+x0 */
      x=x2-y2+x0;
      /* x2=x*x */
      x2=sq_i128(x,bshift);
      /* y2=y*y */
      y2=sq_i128(y,bshift);
    }
    *((double*)PyArray_GETPTR1((PyArrayObject*)col,j))=1-(double)i/maxiter;
  }
  
  return col;
}

static PyMethodDef Int256MandelMethods[] = {
    {"mandel_column",  mandel_column, METH_VARARGS,
     "Evaluate the Mandelbrot formula for a column using 128 bit ints"
     " and 256 bit products."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef int256_mandel = {
    PyModuleDef_HEAD_INIT,
    "int256_mandel",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    Int256MandelMethods
};

PyMODINIT_FUNC
PyInit_int256_mandel(void)
{
  import_array();
  return PyModule_Create(&int256_mandel);
}
