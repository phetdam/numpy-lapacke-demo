/**
 * @file mnewtoninternal.h
 * @author Derek Huang <djh458@stern.nyu.edu>
 * @brief Header to let extensions use internal C functions in `_mnewton.c`.
 */

#ifndef NPY_LPK_MNEWTONINTERNAL_H
#define NPY_LPK_MNEWTONINTERNAL_H

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
#define Py__mnewton_API_pointers 7

// API indices for each of the exposed C functions from _mnewton.c
#define Py__mnewton_remove_specified_kwargs_NUM 0
#define Py__mnewton_remove_unspecified_kwargs_NUM 1
#define Py__mnewton_npy_frob_norm_NUM 2
#define Py__mnewton_tuple_prepend_single_NUM 3
#define Py__mnewton_compute_loss_grad_NUM 4
#define Py__mnewton_compute_hessian_NUM 5
#define Py__mnewton_populate_OptimizeResult_NUM 6

// in client modules, define the void ** API and the import function.
#ifndef MNEWTON_MODULE
static void **Py__mnewton_API;
// internal C functions from _mnewton.c
#define Py__mnewton_remove_specified_kwargs \
  (*(Py_ssize_t (*)(PyObject *, const char **, int)) \
  Py__mnewton_API[Py__mnewton_remove_specified_kwargs_NUM])
#define Py__mnewton_remove_unspecified_kwargs \
  (*(Py_ssize_t (*)(PyObject *, const char **, int)) \
  Py__mnewton_API[Py__mnewton_remove_unspecified_kwargs_NUM])
#define Py__mnewton_npy_frob_norm \
  (*(double (*)(PyArrayObject *)) \
  Py__mnewton_API[Py__mnewton_npy_frob_norm_NUM])
#define Py__mnewton_tuple_prepend_single \
  (*(PyTupleObject *(*)(PyObject *, PyTupleObject *)) \
  Py__mnewton_API[Py__mnewton_tuple_prepend_single_NUM])
#define Py__mnewton_compute_loss_grad \
  (*(PyTupleObject *(*)(PyObject *, PyObject *, PyTupleObject *)) \
  Py__mnewton_API[Py__mnewton_compute_loss_grad_NUM])
#define Py__mnewton_compute_hessian \
  (*(PyArrayObject *(*)(PyObject *, PyTupleObject *)) \
  Py__mnewton_API[Py__mnewton_compute_hessian_NUM])
#define Py__mnewton_populate_OptimizeResult \
  (*(PyObject *(*)(PyArrayObject *, int, int, const char *, PyObject *, \
  PyArrayObject *, PyArrayObject *, PyArrayObject *, \
  Py_ssize_t, Py_ssize_t, Py_ssize_t, Py_ssize_t, PyObject *)) \
  Py__mnewton_API[Py__mnewton_populate_OptimizeResult_NUM])

/**
 * Makes the `_mnewton.c` C API available.
 * 
 * Attempts to import from full path, current directory, and relative path.
 * 
 * @returns `-1` on failure, `0` on success.
 */
static int
_import__mnewton(void)
{
  Py__mnewton_API = (void **) PyCapsule_Import(
    "npy_lapacke_demo.solvers._mnewton._C_API", 0
  );
  return (Py__mnewton_API == NULL) ? -1 : 0;
}

#define import__mnewton() \
  { \
    if (_import__mnewton() < 0) { \
      PyErr_SetString(PyExc_ImportError, "could not import _mnewton C API"); \
      return NULL; \
    } \
  }
#endif /* MNEWTON_MODULE */

#endif /* NPY_LPK_MNEWTONINTERNAL_H */