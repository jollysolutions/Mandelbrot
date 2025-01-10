/* Calculate a column of the Mandelbrot Set, using GMP and numpy
 * and interfacing with python. This version does not use GMP
 * in the python part.
 *
 * MJR 7/2022
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <gmp.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

/* Convert a PyLong to an GMP MPZ.
 * Return 0 on success, -1 on failure
 */
static int PyLong2GMP(mpz_t out, PyLongObject *in){
  int i,digits,sign,is_neg;
  mpz_t tmp;

  if (!PyLong_Check(in)) return -1;

  sign=1;
  is_neg=-1;
  digits=in->long_value.lv_tag >> 3;
  if (digits==0){
    mpz_set_ui(out,(long)0);
    return 0;
  }

  if (digits<0){
    sign=-1;
    digits=-digits;
  }
  is_neg=(in->long_value.lv_tag & 3) == 2;

  mpz_set_ui(out,in->long_value.ob_digit[0]);

  if (digits==1) {
    if (sign==-1) mpz_neg(out,out);
    if (is_neg) mpz_neg(out,out);
    return 0;
  }

  mpz_init2(tmp,digits*PyLong_SHIFT);
  mpz_realloc2(out,digits*PyLong_SHIFT);

  for(i=1;i<digits;i++){
    mpz_set_ui(tmp,in->long_value.ob_digit[i]);
    mpz_mul_2exp(tmp,tmp,i*PyLong_SHIFT);
    mpz_add(out,out,tmp);
  }

  if (sign==-1) mpz_neg(out,out);

  return 0;
}

static PyObject *mandel_column(PyObject *self, PyObject *args){
  int size,maxiter;
  long bshift;
  PyObject *x0_py,*ymin_py,*ymax_py,*col;
  mpz_t x,y,x0,y0,ymin,ymax,x2,y2,mod_test,tmp;
  int i,j;
  npy_intp dims[1];

  if (!PyArg_ParseTuple(args,"OOOiil",&x0_py,&ymin_py,&ymax_py,
			&size,&maxiter,&bshift)){
    fprintf(stderr,"Argument error\n");
    exit(1);
  }

  dims[0]=size;
  col=PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  mpz_inits(x,y,x0,y0,ymin,ymax,x2,y2,mod_test,tmp,NULL);

  if (PyLong2GMP(x0,(PyLongObject*)x0_py)){
    fprintf(stderr,"Error: 1st argument to mandel_column not of type PyLong\n");
    exit(1);
  }
  if (PyLong2GMP(ymin,(PyLongObject*)ymin_py)){
    fprintf(stderr,"Error: 1st argument to mandel_column not of type PyLong\n");
    exit(1);
  }
  if (PyLong2GMP(ymax,(PyLongObject*)ymax_py)){
    fprintf(stderr,"Error: 1st argument to mandel_column not of type PyLong\n");
    exit(1);
  }

  /* mod_test = 4<<(2*bshift) */
  mpz_set_ui(mod_test,(unsigned long)4);
  mpz_mul_2exp(mod_test,mod_test,2*bshift);

  for(j=0;j<size;j++){

    /* y0=ymax-int((j*(ymax-ymin))/size) */
    mpz_sub(y0,ymax,ymin);
    mpz_mul_ui(y0,y0,(unsigned long)j);
    mpz_tdiv_q_ui(y0,y0,(unsigned long)size);
    mpz_sub(y0,ymax,y0);

    mpz_set(x,x0);
    mpz_set(y,y0);
    mpz_mul(x2,x,x);
    mpz_mul(y2,y,y);

    for(i=0;i<maxiter;i++){
      /* tmp=x2+y2 */
      mpz_add(tmp,x2,y2);
      if (mpz_cmp(tmp,mod_test)>=0) break;
      /* y=((x*y)>>(bshift-1))+y0 */
      mpz_mul(y,x,y);
      mpz_tdiv_q_2exp(y,y,bshift-1);
      mpz_add(y,y,y0);
      /* x=((x2-y2)>>bshift)+x0 */
      mpz_sub(x,x2,y2);
      mpz_tdiv_q_2exp(x,x,bshift);
      mpz_add(x,x,x0);
      /* x2=x*x */
      mpz_mul(x2,x,x);
      /* y2=y*y */
      mpz_mul(y2,y,y);
    }
    *((double*)PyArray_GETPTR1((PyArrayObject*)col,j))=1-(double)i/maxiter;
  }

  mpz_clears(x,y,x0,y0,ymin,ymax,x2,y2,mod_test,tmp,NULL);

  return col;
}

static PyMethodDef GmpMandelMethods[] = {
    {"mandel_column",  mandel_column, METH_VARARGS,
     "Evaluate the Mandelbrot formula for a column using GMP."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gmp_mandel = {
    PyModuleDef_HEAD_INIT,
    "gmp_mandel",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    GmpMandelMethods
};

PyMODINIT_FUNC
PyInit_gmp_mandel(void)
{
  import_array();
  return PyModule_Create(&gmp_mandel);
}
