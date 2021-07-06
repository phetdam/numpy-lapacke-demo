/**
 * @file linreginternal.h
 * @author Derek Huang <djh458@stern.nyu.edu>
 * @brief Header to let extensions use internal C functions in `_linreg.c`.
 */

#ifndef NPY_LPK_LINREGINTERNAL_H
#define NPY_LPK_LINREGINTERNAL_H

// PY_SSIZE_T_CLEAN, NPY_NO_DEPRECATED_API defines must be guarded
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif /* PY_SSIZE_T_CLEAN */

#include <Python.h>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif /* NPY_NO_DEPRECATED_API */

#include <numpy/arrayobject.h>

// total number of function pointers stored in the void ** C API
#define Py__linreg_API_pointers 2

// API indices for each of the exposed C functions from _linreg.c
#define Py__linreg_npy_vector_matrix_mean_NUM 0
#define Py__linreg_compute_intercept_NUM 1

// in client modules, define the void ** API and the import function.
#ifndef LINREG_MODULE
static void **Py__linreg_API;
// internal C functions from _linreg.c
#define Py__linreg_npy_vector_matrix_mean \
  (*(PyObject *(*)(PyArrayObject *)) \
  Py__linreg_API[Py__linreg_npy_vector_matrix_mean_NUM])
#define Py__linreg_compute_intercept \
  (*(PyObject *(*)(PyArrayObject *, PyArrayObject *, PyObject *)) \
  Py__linreg_API[Py__linreg_compute_intercept_NUM])

/**
 * Makes the `_linreg.c` C API available.
 * 
 * Attempts to import from full path, current directory, and relative path.
 * 
 * @returns `-1` on failure, `0` on success.
 */
static int
_import__linreg(void)
{
  Py__linreg_API = (void **) PyCapsule_Import(
    "npy_lapacke_demo.regression._linreg._C_API", 0
  );
  return (Py__linreg_API == NULL) ? -1 : 0;
}

#define import__linreg() \
  { \
    if (_import__linreg() < 0) { \
      PyErr_SetString(PyExc_ImportError, "could not import _linreg C API"); \
      return NULL; \
    } \
  }
#endif /* LINREG_MODULE */

#endif /* NPY_LPK_LINREGINTERNAL_H */