/* Calculate a column of the Mandelbrot Set, using GMP and numpy
 * and interfacing with python. This version does not use GMP
 * in the python part.
 *
 * MJR 7/2022
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

static PyObject *mandel_column(PyObject *self, PyObject *args){
  int size,maxiter;
  long bshift;
  PyObject *x0_py,*ymin_py,*ymax_py,*col;
  int64_t x,y,x0,y0,ymin,ymax;
  __int128 x2,y2,mod_test,tmp;
  int i,j;
  npy_intp dims[1];
  
  if (!PyArg_ParseTuple(args,"OOOiil",&x0_py,&ymin_py,&ymax_py,
			&size,&maxiter,&bshift)){
    fprintf(stderr,"Argument error\n");
    exit(1);
  }
  
  dims[0]=size;
  col=PyArray_SimpleNew(1,dims,NPY_DOUBLE);
  
  x0=PyLong_AsLongLong(x0_py);
  ymin=PyLong_AsLongLong(ymin_py);
  ymax=PyLong_AsLongLong(ymax_py);

  mod_test=4;
  mod_test<<=(2*bshift);

  for(j=0;j<size;j++){

    y0=ymax-((j*(__int128)(ymax-ymin))/size);

    x=x0;
    y=y0;
    x2=((__int128)x)*x;
    y2=((__int128)y)*y;

    for(i=0;i<maxiter;i++){
      /* tmp=x2+y2 */
      if (x2+y2>=mod_test) break;
      /* y=((x*y)>>(bshift-1))+y0 */
      tmp=((__int128)x)*y;
      if (tmp>=0)
	tmp=tmp>>(bshift-1);
      else
	tmp=-((-tmp)>>(bshift-1));
      y=(int64_t)tmp+y0;
      /* x=((x2-y2)>>bshift)+x0 */
      tmp=x2-y2;
      if (tmp>=0)
	tmp=tmp>>bshift;
      else
	tmp=-((-tmp)>>bshift);
      x=(int64_t)tmp+x0;
      /* x2=x*x */
      x2=((__int128)x)*x;
      /* y2=y*y */
      y2=((__int128)y)*y;
    }
    *((double*)PyArray_GETPTR1((PyArrayObject*)col,j))=1-(double)i/maxiter;
  }
  
  return col;
}

static PyMethodDef Int128MandelMethods[] = {
    {"mandel_column",  mandel_column, METH_VARARGS,
     "Evaluate the Mandelbrot formula for a column using 64 and 128 bit ints."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef int128_mandel = {
    PyModuleDef_HEAD_INIT,
    "int128_mandel",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    Int128MandelMethods
};

PyMODINIT_FUNC
PyInit_int128_mandel(void)
{
  import_array();
  return PyModule_Create(&int128_mandel);
}
