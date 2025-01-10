
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <gmp.h>
#include <gmpy2.h>

/* Next two defs from /usr/lib/python3/dist-packages/gmpy2/gmpy2.h
 * So if you can't find gmpy2.h, comment out the include line, and
 * change the 0 to a 1 on the line below...
 */

#if 0
typedef struct {
    PyObject_HEAD
    mpz_t z;
    Py_hash_t hash_cache;
} MPZ_Object;

#define MPZ(obj)  (((MPZ_Object*)(obj))->z)
#endif

static PyObject *mandel(PyObject *self, PyObject *args){
  int maxiter;
  long bshift;
  PyObject *x0_py,*y0_py;
  mpz_t x,y,x0,y0,x2,y2,mod_test,tmp;
  int i;

  if (!PyArg_ParseTuple(args,"OOli",&x0_py,&y0_py,&bshift,&maxiter)){
    fprintf(stderr,"Argument error\n");
    exit(1);
  }

  mpz_inits(x,y,x0,y0,x2,y2,mod_test,tmp,NULL);

  mpz_set(x0,MPZ(x0_py));
  mpz_set(y0,MPZ(y0_py));

  /* mod_test = 4<<(2*bshift) */
  mpz_set_ui(mod_test,(unsigned long)4);
  mpz_mul_2exp(mod_test,mod_test,2*bshift);

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

  mpz_clears(x,y,x0,y0,x2,y2,mod_test,tmp,NULL);

  return PyLong_FromLong((long)i);

}

static PyMethodDef GmpMandelMethods[] = {
    {"mandel",  mandel, METH_VARARGS,
     "Evaluate the Mandelbrot formula for a point using GMP."},
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
    import_gmpy2();
    return PyModule_Create(&gmp_mandel);
}
