/**
 * @file _mnewton.c
 * @brief C implementation of Newton's method with Hessian modification.
 * 
 * The modified Hessian is the original Hessian with an added multiple of the
 * identity, checked to be positive definite by Cholesky decomposition. Method
 * is compatible as a frontend for scipy.optimize.minimize.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <limits.h>
#include <string.h>
#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// npt_lapacke_demo/*.h automatically handles the different includes
// depending on whether Intel MKL, OpenBLAS, or system CBLAS/LAPACKE is linked
#include "npy_lapacke_demo/cblas.h"
#include "npy_lapacke_demo/lapacke.h"
// defines the EXPOSE_INTERNAL_NOTICE macro
#include "npy_lapacke_demo/extutils.h"

// docstring for mnewton
PyDoc_STRVAR(
  mnewton_doc,
  "mnewton(fun, x0, args=(), jac=None, hess=None, tol=1e-4, maxiter=1000,\n"
  "alpha=0.5, beta=1e-3, gamma=0.8, **ignored)"
  "\n--\n\n"
  "Newton's method, adding scaled identity matrix to the Hessian if necessary."
  "\n\n"
  "Whenever the Hessian is not positive definite, a multiple of the identity\n"
  "is added to the Hessian until the Cholesky decomposition of the modified\n"
  "Hessian succeeds, at which point the descent direction is computed. Step\n"
  "size is chosen using a simple backtracking line search implementation,\n"
  "where the chosen step satisfies the Armijo condition."
  "\n\n"
  "Hessian modification implements Algorithm 3.3 in [#]_, on page 51."
  "\n\n"
  ".. [#] Nocedal, J., & Wright, S. (2006). *Numerical Optimization*.\n"
  "   Springer Science and Business Media."
);
/**
 * Newton's method with Hessian modification.
 * 
 * Uses Cholesky decomposition to check if the Hessian plus the scaled identity
 * matrix is positive definite before computing the search direction.
 * 
 * @param self `PyObject *` module (unused)
 * @param args `PyObject *` positional args tuple
 * @param kwargs `PyObject *` keyword args dict
 * @returns `scipy.optimize.OptimizeResult`
 */
static PyObject *
mnewton(PyObject *self, PyObject *args, PyObject *kwargs)
{
  Py_RETURN_NONE;
}

/**
 * Returns a new `scipy.optimize.OptimizeResult` with some fields set.
 * 
 * Remove from production release!
 * 
 * @param self `PyObject *` module (unused)
 * @param args `PyObject *` positional args (unused)
 * @returns New reference to a `scipy.optimize.OptimizeResult` instance on
 *     success, otherwise `NULL` with exception set on error.
 */
static PyObject *
new_OptimizeResult(PyObject *self, PyObject *args)
{
  // import scipy.optimize/get a new reference if already imported
  PyObject *spopt = PyImport_ImportModule("scipy.optimize");
  if (spopt == NULL) {
    return NULL;
  }
  // get the OptimizeResult member
  PyObject *OptimizeResult = PyObject_GetAttrString(spopt, "OptimizeResult");
  if (OptimizeResult == NULL) {
    goto except_spopt;
  }
  // arbitrary ndarray to represent a return result, flags NPY_ARRAY_CARRAY
  npy_intp dims[] = {10};
  PyArrayObject *res_ar;
  res_ar = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  if (res_ar == NULL) {
    goto except_OptimizeResult;
  }
  // call OptimizeResult with no args to get an empty OptimizeResult
  PyObject *res_obj = PyObject_CallObject(OptimizeResult, NULL);
  if (res_obj == NULL) {
    goto except_res_ar;
  }
  // get data pointer of res and fill array with some values
  double *res_data = (double *) PyArray_DATA(res_ar);
  for (npy_intp i = 0; i < PyArray_DIM(res_ar, 0); i++) {
    res_data[i] = (1782 + 11 * i) % 10;
  }
  // optimization status (set to 0 for successful exit)
  PyObject *res_status = PyLong_FromLong(0);
  if (res_status == NULL) {
    goto except_res_ar;
  }
  // message for the cause of termination (successful)
  PyObject *res_message = PyUnicode_FromString("optimization successful");
  if (res_message == NULL) {
    goto except_res_status;
  }
  // add res_ar, Py_True, res_status, res_message to res_obj. note that Py_True
  // needs to have its reference count incremented.
  if (PyObject_SetAttrString(res_obj, "x", (PyObject *) res_ar) < 0) {
    goto except_res_message;
  }
  Py_INCREF(Py_True);
  if (PyObject_SetAttrString(res_obj, "success", Py_True) < 0) {
    // no  goto label for Py_True since we will Py_DECREF it on success
    Py_DECREF(Py_True);
    goto except_res_message;
  }
  Py_DECREF(Py_True);
  if (PyObject_SetAttrString(res_obj, "status", res_status) < 0) {
    goto except_res_message;
  }
  if (PyObject_SetAttrString(res_obj, "message", res_message) < 0) {
    goto except_res_message;
  }
  // clean up and return res_obj
  Py_DECREF(res_message);
  Py_DECREF(res_status);
  Py_DECREF(res_ar);
  Py_DECREF(OptimizeResult);
  Py_DECREF(spopt);
  return res_obj;
// clean up on exceptions
except_res_message:
  Py_DECREF(res_message);
except_res_status:
  Py_DECREF(res_status);
except_res_ar:
  Py_DECREF(res_ar);
except_OptimizeResult:
  Py_DECREF(OptimizeResult);
except_spopt:
  Py_DECREF(spopt);
  return NULL;
}

// _mnewton methods, possibly including EXTERNAL_* wrappers
static PyMethodDef _mnewton_methods[] = {
  {
    "mnewton", (PyCFunction) mnewton,
    METH_VARARGS | METH_KEYWORDS, mnewton_doc
  },
  {
    "new_OptimizeResult", (PyCFunction) new_OptimizeResult,
    METH_VARARGS, NULL
  },
// make EXPOSED_* methods accessible if EXPOSE_INTERNAL defined.
// __INTELLISENSE__ always defined in VS Code; allows Intellisense to work.
#if defined(__INTELLISENSE__) || defined(EXPOSE_INTERNAL)
#endif /* EXPOSE_INTERNAL */
  // sentinel marking end of array
  {NULL, NULL, 0, NULL}
};

// _mnewton module docstring
PyDoc_STRVAR(
  _mnewton_doc,
  "C implementation of Newton's method with Hessian modification."
  "\n\n"
  "The modified Hessian is the original Hessian with an added multiple of\n"
  "the identity, checked to be positive definite by Cholesky decomposition.\n"
  "Method ``mnewton`` can be used as a frontend for scipy.optimize.minimize."
);
// _mnewton module definition
static PyModuleDef _mnewton_module = {
  PyModuleDef_HEAD_INIT,
  // name, docstring, size = -1 to disable subinterpreter support, methods
  .m_name = "_mnewton",
  .m_doc = _mnewton_doc,
  .m_size = -1,
  .m_methods = _mnewton_methods
};

// module initialization function
PyMODINIT_FUNC
PyInit__mnewton(void)
{
  // import NumPy Array C API. automatically returns NULL on error.
  import_array();
  // create module and return. NULL on error
  return PyModule_Create(&_mnewton_module);
}