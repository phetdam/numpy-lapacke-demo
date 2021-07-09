/**
 * @file _mnewton_internal.c
 * @author Derek Huang <djh458@stern.nyu.edu>
 * @brief Wrappers for internal C functions in `_mnewton.c`.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// gives access to internal C functions in _mnewton.c
#include "mnewtoninternal.h"

// flags to pass to remove_kwargs_dispatcher controlling whether to call
// remove_specified_kwargs or remove_unspecified_kwargs
#define REMOVE_KWARGS_SPECIFIED 0
#define REMOVE_KWARGS_UNSPECIFIED 1 
// argument names known to remove_kwargs_dispatcher
static const char *remove_kwargs_dispatcher_argnames[] = {
  "kwargs", "kwargs_list", "warn", NULL
};
/**
 * Internal `remove_specified_kwargs`, `remove_unspecified_kwargs` dispatcher.
 * 
 * Since both functions have the same signatures and require the same input
 * checking, their respective wrappers can just wrap this function.
 * 
 * @param args `PyObject *` tuple of positional args
 * @param kwargs `PyObject *` giving any keyword arguments, may be `NULL`
 * @param dispatch_flag `int` indicating whether to call
 *     `remove_specified_kwargs` or `remove_unspecified_kwargs`. Must only be
 *     `REMOVE_KWARGS_SPECIFIED` or `REMOVE_KWARGS_UNSPECIFIED`.
 * @returns New reference to a `PyLongObject` giving the number of names
 *     dropped from the kwargs dict on success, `NULL` on failure.
 */
static PyObject *
remove_kwargs_dispatcher(PyObject *args, PyObject *kwargs, int dispatch_flag)
{
  // kwargs dict, kwargs_list (may corespond to droplist or keeplist)
  PyObject *kwdict, *py_kwargs_list;
  kwdict = py_kwargs_list = NULL;
  // whether to warn or not
  int warn = 1;
  // dispatch_flag must be REMOVE_KWARGS_SPECIFIED or REMOVE_KWARGS_UNSPECIFIED
  if (
    dispatch_flag != REMOVE_KWARGS_SPECIFIED &&
    dispatch_flag != REMOVE_KWARGS_UNSPECIFIED
  ) {
    PyErr_SetString(
      PyExc_RuntimeError, "dispatch_flag must only be "
      "REMOVE_KWARGS_SPECIFIED or REMOVE_KWARGS_UNSPECIFIED"
    );
    return NULL;
  }
  // parse arguments
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O!O!p", (char **) remove_kwargs_dispatcher_argnames,
      &PyDict_Type, &kwdict, &PyList_Type, &py_kwargs_list, &warn
    )
  ) {
    return NULL;
  }
  // get length of dict. remove_specified_kwargs works with empty dicts.
  Py_ssize_t n_kwds = PyDict_Size(kwdict);
  // get keys of kwdict if dict is not empty
  PyObject *keys;
  // if there are keys in the dictionary, we have to check them
  if (n_kwds > 0) {
    keys = PyDict_Keys(kwdict);
    if (keys == NULL) {
      return NULL;
    }
    // for each item in the list of keys
    for (Py_ssize_t i = 0; i < n_kwds; i++) {
      // if ith item is not a PyUnicode (exact), error
      if (!PyUnicode_CheckExact(PyList_GET_ITEM(keys, i))) {
        Py_DECREF(keys);
        return NULL;
      }
    }
    // if all keys are string keys, done; don't need new reference
    Py_DECREF(keys);
  }
  // number of elements in py_kwargs_list (no need to error check)
  Py_ssize_t n_drop = PyList_GET_SIZE(py_kwargs_list);
  // if empty, raise error. message differs depending on dispatch_flag
  if (n_drop == 0) {
    if (dispatch_flag == REMOVE_KWARGS_SPECIFIED) {
      PyErr_SetString(PyExc_ValueError, "droplist must be nonempty");
    }
    else {
      PyErr_SetString(PyExc_ValueError, "keeplist must be nonempty");
    }
    return NULL;
  }
  // check that all elements of py_kwargs_list are string as well
  for (Py_ssize_t i = 0; i < n_drop; i++) {
    if (!PyUnicode_CheckExact(PyList_GET_ITEM(py_kwargs_list, i))) {
      return NULL;
    }
  }
  // create array of strings from py_kwargs_list. +1 for ending NULL
  const char **kwargs_list = (const char **) PyMem_RawMalloc(
    (size_t) (n_drop + 1) * sizeof(char **)
  );
  if (kwargs_list == NULL) {
    return NULL;
  }
  // set the last member to NULL and populate const char * in kwargs_list
  kwargs_list[n_drop] = NULL;
  for (Py_ssize_t i = 0; i < n_drop; i++) {
    kwargs_list[i] = PyUnicode_AsUTF8(PyList_GET_ITEM(py_kwargs_list, i));
    // kwargs_list[i] is NULL on error, so we have to clean up kwargs_list
    if (kwargs_list[i] == NULL) {
      goto except;
    }
  }
  /**
   * pass kwdict, kwargs_list, warn to Py__mnewton_remove_specified_kwargs or
   * Py__mnewton_remove_unspecified_kwargs depending on dispatch_flag and save
   * the number of keys dropped. drops will be -1 on error.
   */
  Py_ssize_t drops;
  if (dispatch_flag == REMOVE_KWARGS_SPECIFIED) {
    drops = Py__mnewton_remove_specified_kwargs(kwdict, kwargs_list, warn);
  }
  else {
    drops = Py__mnewton_remove_unspecified_kwargs(kwdict, kwargs_list, warn);
  }
  if (drops < 0) {
    goto except;
  }
  // clean up kwargs_list and return PyLong from drops (NULL on error)
  PyMem_RawFree(kwargs_list);
  return PyLong_FromSsize_t(drops);
// clean up on error
except:
  PyMem_RawFree(kwargs_list);
  return NULL;
}

// docstring for remove_specified_kwargs wrapper
PyDoc_STRVAR(
  remove_specified_kwargs_doc,
  "remove_specified_kwargs(kwargs, droplist, warn=True)"
  "\n--\n\n"
  "Python-accessible wrapper for internal function remove_specified_kwargs."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "kwargs : dict\n"
  "    dict containing str keys only, representing the kwargs dict often\n"
  "    unpacked to provide named arguments to functions.\n"
  "droplist : list\n"
  "    List of strings indicating which names in kwargs to drop.\n"
  "warn : bool, default=True\n"
  "    True to warn if a name in droplist is not in kwargs, False to not\n"
  "    warn if a name in droplist is not in kwargs."
  "\n\n"
  "Returns\n"
  "-------\n"
  "int\n"
  "    The number of names in droplist dropped from kwargs."
);
/**
 * Python-accessible wrapper for `remove_specified_kwargs`.
 * 
 * @param self `PyObject *` module (unused)
 * @param args `PyObject *` tuple of positional args
 * @param kwargs `PyObject *` giving any keyword arguments, may be `NULL`
 * @returns New reference to `PyLongObject *` on success, `NULL` on failure.
 */
static PyObject *
remove_specified_kwargs(PyObject *self, PyObject *args, PyObject *kwargs)
{
  // simply wrap remove_kwargs_dispatcher
  return remove_kwargs_dispatcher(args, kwargs, REMOVE_KWARGS_SPECIFIED);
}

// docstring for remove_unspecified_kwargs wrapper
PyDoc_STRVAR(
  remove_unspecified_kwargs_doc,
  "remove_unspecified_kwargs(kwargs, droplist, warn=True)"
  "\n--\n\n"
  "Python-accessible wrapper for internal remove_unspecified_kwargs."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "kwargs : dict\n"
  "    dict containing str keys only, representing the kwargs dict often\n"
  "    unpacked to provide named arguments to functions.\n"
  "keeplist : list\n"
  "    List of strings indicating which names in kwargs to keep.\n"
  "warn : bool, default=True\n"
  "    True to warn if a name not in keeplist has been removed from kwargs,\n"
  "    False to otherwise not issue the warning."
  "\n\n"
  "Returns\n"
  "-------\n"
  "int\n"
  "    The number of names in dropped from kwargs not in keeplist."
);
/**
 * Python-accessible wrapper for `remove_specified_kwargs`.
 * 
 * @param self `PyObject *` module (unused)
 * @param args `PyObject *` tuple of positional args
 * @param kwargs `PyObject *` giving any keyword arguments, may be `NULL`
 * @returns New reference to `PyLongObject *` on success, `NULL` on failure.
 */
static PyObject *
remove_unspecified_kwargs(PyObject *self, PyObject *args, PyObject *kwargs)
{
  // simply wrap remove_kwargs_dispatcher
  return remove_kwargs_dispatcher(args, kwargs, REMOVE_KWARGS_UNSPECIFIED);
}

// docstring for npy_frob_norm wrapper
PyDoc_STRVAR(
  npy_frob_norm_doc,
  "npy_frob_norm(ar)"
  "\n--\n\n"
  "Python-accessible wrapper for internal functon npy_frob_norm."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "ar : numpy.ndarray\n"
  "    NumPy array with flags NPY_ARRAY_ALIGNED, type NPY_DOUBLE, or at\n"
  "    least any object that can be converted to such a NumPy array."
  "\n\n"
  "Returns\n"
  "-------\n"
  "float\n"
  "    Frobenius norm of the NumPy array."
);
/**
 * Python-accessible wrapper for internal function `npy_frob_norm`.
 * 
 * @param self `PyObject *` module (unused)
 * @param arg `PyObject *` single positional arg
 * @returns New reference to `PyFloatObject` giving the norm of the NumPy
 *     array on success, `NULL` with set exception on failure.
 */
static PyObject *
npy_frob_norm(PyObject *self, PyObject *arg)
{
  // convert arg to ndarray with NPY_DOUBLE type and NPY_ARRAY_ALIGNED flags
  PyArrayObject *ar;
  ar = (PyArrayObject *) PyArray_FROM_OTF(arg, NPY_DOUBLE, NPY_ARRAY_ALIGNED);
  if (ar == NULL) {
    return NULL;
  }
  // return PyFloatObject * from npy_frob_norm, NULL on error
  return (PyObject *) PyFloat_FromDouble(Py__mnewton_npy_frob_norm(ar));
}

// docstring for tuple_prepend_single wrapper
PyDoc_STRVAR(
  tuple_prepend_single_doc,
  "tuple_prepend_single(x, old_tp=None)"
  "\n--\n\n"
  "Python-accessible wrapper for internal functon tuple_prepend_single."
  "\n\n"
  "Equivalent to returning (x, *old_tp) in Python."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "x : object\n"
  "    Arbitrary Python object.\n"
  "old_tp : tuple, default=None\n"
  "    A Python tuple. If not provided, then (x,) is returned."
  "\n\n"
  "Returns\n"
  "-------\n"
  "tuple"
);
// arguments known to tuple_prepend_single
static const char *tuple_prepend_single_argnames[] = {
  "x", "old_tp", NULL
};
/**
 * Python-accessible wrapper for internal function `tuple_prepend_single`.
 * 
 * @param self `PyObject *` module (unused)
 * @param args `PyObject *` tuple of positional args
 * @param kwargs `PyObject *` dict of keyword arguments, may be `NULL`
 * @returns New reference to `PyTupleObject`, `NULL` on error.
 */
static PyObject *
tuple_prepend_single(PyObject *self, PyObject *args, PyObject *kwargs)
{
  // x, old_tp. old_tp must be NULL since it may not be modified
  PyObject *x;
  PyTupleObject *old_tp = NULL;
  // parse arguments
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|O!", (char **) tuple_prepend_single_argnames,
      &x, &PyTuple_Type, &old_tp
    )
  ) {
    return NULL;
  }
  // return result from Py__mnewton_tuple_prepend_single (NULL on error)
  return (PyObject *) Py__mnewton_tuple_prepend_single(x, old_tp);
}

// docstring for loss_only_fun_call wrapper
PyDoc_STRVAR(
  loss_only_fun_call_doc,
  "loss_only_fun_call(fun, x, args=None)"
  "\n--\n\n"
  "Python-accessible wrapper for internal function loss_only_fun_call."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "fun : function\n"
  "    Objective function with signature fun(x, *args). Must return a float\n"
  "    or something that can be converted to a float, else (loss, grad).\n"
  "x : numpy.ndarray\n"
  "    Point to evaluate fun at. Must have type NPY_DOUBLE and flags\n"
  "    NPY_ARRAY_IN_ARRAY or be convertible to such n array. x must have\n"
  "    shape (n_features,) and checks will be performed to ensure this.\n"
  "args : tuple, default=None\n"
  "    Additional positional arguments to pass to fun, jac."
  "\n\n"
  "Returns\n"
  "-------\n"
  "float\n"
  "    Current value of the objective function at x\n"
);
// argument names known to loss_only_fun_call
static const char *loss_only_fun_call_argnames[] = {"fun", "x", "args", NULL};
/**
 * Python-accessible wrapper for `loss_only_fun_call`.
 * 
 * @param self `PyObject *` module (unused)
 * @param args `PyObject *` tuple of positional args
 * @param kwargs `PyObject *` giving any keyword arguments, may be `NULL`
 * @returns New reference to `PyTupleObject *` on success, `NULL` on failure.
 */
static PyObject *
loss_only_fun_call(PyObject *self, PyObject *args, PyObject *kwargs)
{
  // objective, result, x (parameter guess)
  PyObject *fun, *res;
  PyArrayObject *x;
  // positional args for fun containing x as first element. fun_args must be
  // NULL since it may not be modified by PyArg_ParseTupleAndKeywords
  PyTupleObject *fun_args;
  fun_args = NULL;
  // parse arguments using PyArg_ParseTupleAndKeywords
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "OO|O!", (char **) loss_only_fun_call_argnames,
      &fun, &x, &PyTuple_Type, &fun_args
    )
  ) {
    return NULL;
  }
  // success. check that fun is callable
  if (!PyCallable_Check(fun)) {
    PyErr_SetString(PyExc_TypeError, "fun must be callable");
    return NULL;
  }
  // convert x to numpy.ndarray with NPY_DOUBLE type, NPY_ARRAY_IN_ARRAY flags
  x = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  if (x == NULL) {
    return NULL;
  }
  // check that x is nonempty and 1D array
  if (PyArray_SIZE(x) == 0) {
    PyErr_SetString(PyExc_ValueError, "x must be nonempty");
    goto except_x;
  }
  if (PyArray_NDIM(x) != 1) {
    PyErr_SetString(PyExc_ValueError, "x must be 1D array");
    goto except_x;
  }
  // use Py__mnewton_tuple_prepend_single to get fun_args. NULL on error. it's
  // fine to drop the borrowed reference.
  fun_args = Py__mnewton_tuple_prepend_single((PyObject *) x, fun_args);
  if (fun_args == NULL) {
    goto except_x;
  }
  // call Py__mnewton_loss_only_fun_call and get res (NULL on error)
  res = Py__mnewton_loss_only_fun_call(fun, fun_args);
  if (res == NULL) {
    goto except_fun_args;
  }
  // if no problems, clean up and return
  Py_DECREF(fun_args);
  Py_DECREF(x);
  return (PyObject *) res;
// clean up on errors
except_fun_args:
  Py_DECREF(fun_args);
except_x:
  Py_DECREF(x);
  return NULL;
}

// docstring for compute_loss_grad wrapper
PyDoc_STRVAR(
  compute_loss_grad_doc,
  "compute_loss_grad(fun, jac, x, args=None)"
  "\n--\n\n"
  "Python-accessible wrapper for internal function compute_loss_grad."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "fun : function\n"
  "    Objective function with signature fun(x, *args). Must return a float\n"
  "    or something that can be converted to a float.\n"
  "jac : function or True\n"
  "    Gradient function with signature jac(x, *args) is callable. May be\n"
  "    True, in which case fun must return (loss, grad).\n"
  "x : numpy.ndarray\n"
  "    Point to evaluate fun, jac at. Must have type NPY_DOUBLE and flags\n"
  "    NPY_ARRAY_IN_ARRAY or be convertible to such n array. x must have\n"
  "    shape (n_features,) and checks will be performed to ensure this.\n"
  "args : tuple, default=None\n"
  "    Additional positional arguments to pass to fun, jac."
  "\n\n"
  "Returns\n"
  "-------\n"
  "loss : float\n"
  "    Current value of the objective function at x\n"
  "grad : numpy.ndarray\n"
  "    Current value of the gradient at x, same shape as x"
);
// argument names known to compute_loss_grad
static const char *compute_loss_grad_argnames[] = {
  "fun", "jac", "x", "args", NULL
};
/**
 * Python-accessible wrapper for `compute_loss_grad`.
 * 
 * @param self `PyObject *` module (unused)
 * @param args `PyObject *` tuple of positional args
 * @param kwargs `PyObject *` giving any keyword arguments, may be `NULL`
 * @returns New reference to `PyTupleObject *` on success, `NULL` on failure.
 */
static PyObject *
compute_loss_grad(PyObject *self, PyObject *args, PyObject *kwargs)
{
  // objective, gradient, x (parameter guess)
  PyObject *fun, *jac;
  PyArrayObject *x;
  // positional args for fun, jac containing x as first element, result of
  // compute_loss_grad. fun_args must be NULL since it may not be modified.
  PyTupleObject *fun_args, *res;
  fun_args = NULL;
  // parse arguments using PyArg_ParseTupleAndKeywords
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "OOO|O!", (char **) compute_loss_grad_argnames,
      &fun, &jac, &x, &PyTuple_Type, &fun_args
    )
  ) {
    return NULL;
  }
  // success. check that fun is callable
  if (!PyCallable_Check(fun)) {
    PyErr_SetString(PyExc_TypeError, "fun must be callable");
    return NULL;
  }
  // check that jac is either callable or Py_True
  if (!PyCallable_Check(jac) && jac != Py_True) {
    PyErr_SetString(PyExc_TypeError, "jac must be callable or True");
    return NULL;
  }
  // convert x to numpy.ndarray with NPY_DOUBLE type, NPY_ARRAY_IN_ARRAY flags
  x = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  if (x == NULL) {
    return NULL;
  }
  // check that x is nonempty and 1D array
  if (PyArray_SIZE(x) == 0) {
    PyErr_SetString(PyExc_ValueError, "x must be nonempty");
    goto except_x;
  }
  if (PyArray_NDIM(x) != 1) {
    PyErr_SetString(PyExc_ValueError, "x must be 1D array");
    goto except_x;
  }
  // use Py__mnewton_tuple_prepend_single to get fun_args. NULL on error
  fun_args = Py__mnewton_tuple_prepend_single((PyObject *) x, fun_args);
  if (fun_args == NULL) {
    goto except_x;
  }
  // call Py__mnewton_compute_loss_grad and get res (NULL on error)
  res = Py__mnewton_compute_loss_grad(fun, jac, fun_args);
  if (res == NULL) {
    goto except_fun_args;
  }
  // if no problems, clean up and return
  Py_DECREF(fun_args);
  Py_DECREF(x);
  return (PyObject *) res;
// clean up on errors
except_fun_args:
  Py_DECREF(fun_args);
except_x:
  Py_DECREF(x);
  return NULL;
}

// docstring for compute_hessian wrapper
PyDoc_STRVAR(
  compute_hessian_doc,
  "compute_hessian(hess, x, args=None)"
  "\n--\n\n"
  "Python-accessible wrapper for internal function `compute_hessian`."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "hess : function\n"
  "    Hessian function with signature hess(x, *args). Must return a\n"
  "    numpy.ndarray with type NPY_DOUBLE and flags NPY_ARRAY_CARRAY or\n"
  "    something convertible as such. Returned array should have shape\n"
  "    (n_features, n_features) (not checked).\n"
  "x : numpy.ndarray\n"
  "    Point to evaluate fun, jac at. Must have type NPY_DOUBLE and flags\n"
  "    NPY_ARRAY_IN_ARRAY or be convertible to such n array. x must have\n"
  "    shape (n_features,) and checks will be performed to ensure this.\n"
  "args : tuple, default=None\n"
  "    Additional positional arguments to pass to hess."
  "\n\n"
  "Returns\n"
  "-------\n"
  "numpy.ndarray\n"
  "    Current value of the Hessian function at x"
);
// argument names known to compute_hessian
static const char *compute_hessian_argnames[] = {
  "hess", "x", "args", NULL
};
/**
 * Python-accessible wrapper for `compute_loss_grad`.
 * 
 * @param self `PyObject *` module (unused)
 * @param args `PyObject *` tuple of positional args
 * @param kwargs `PyObject *` giving any keyword arguments, may be `NULL`
 * @returns New reference to `PyArrayObject *` on success, `NULL` on failure.
 */
static PyObject *
compute_hessian(PyObject *self, PyObject *args, PyObject *kwargs)
{
  // hessian, x (parameter guess), result of compute_hessian
  PyObject *hess;
  PyArrayObject *x, *res;
  // positional args for fun, jac containing x as first element. fun_args must
  // be NULL since it may not be modified by PyArg_ParseTupleAndKeywords.
  PyTupleObject *fun_args = NULL;
  // parse arguments using PyArg_ParseTupleAndKeywords
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "OO|O!", (char **) compute_hessian_argnames,
      &hess, &x, &PyTuple_Type, &fun_args
    )
  ) {
    return NULL;
  }
  // success. check that hess is callable
  if (!PyCallable_Check(hess)) {
    PyErr_SetString(PyExc_TypeError, "hess must be callable");
    return NULL;
  }
  // convert x to numpy.ndarray with NPY_DOUBLE type, NPY_ARRAY_IN_ARRAY flags
  x = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  if (x == NULL) {
    return NULL;
  }
  // check that x is nonempty and 1D array
  if (PyArray_SIZE(x) == 0) {
    PyErr_SetString(PyExc_ValueError, "x must be nonempty");
    goto except_x;
  }
  if (PyArray_NDIM(x) != 1) {
    PyErr_SetString(PyExc_ValueError, "x must be 1D array");
    goto except_x;
  }
  // use Py__mnewton_tuple_prepend_single to get fun_args. NULL on error
  fun_args = Py__mnewton_tuple_prepend_single((PyObject *) x, fun_args);
  if (fun_args == NULL) {
    goto except_x;
  }
  // call Py__mnewton_compute_hessian and get res (NULL on error)
  res = Py__mnewton_compute_hessian(hess, fun_args);
  if (res == NULL) {
    goto except_fun_args;
  }
  // if no problems, clean up and return
  Py_DECREF(fun_args);
  Py_DECREF(x);
  return (PyObject *) res;
// clean up on errors
except_fun_args:
  Py_DECREF(fun_args);
except_x:
  Py_DECREF(x);
  return NULL;
}

// docstring for populate_OptimizeResult wrapper
PyDoc_STRVAR(
  populate_OptimizeResult_doc,
  "populate_OptimizeResult(x, success, status, message, fun_x, n_fev,\n"
  "n_iter, jac_x=None, n_jev=None, hess_x=None, n_hev=None,\n"
  "hess_inv=None, maxcv=None)"
  "\n--\n\n"
  "Python-accessible wrapper for internal populate_OptimizeResult."
  "\n\n"
  "Any keyword arguments that are left as None will not be set to attributes\n"
  "in the returned scipy.optimize.OptimizeResult. Unless noted, arguments\n"
  "correspond to their attributes in the OptimizeResult."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "x : numpy.ndarray\n"
  "    Optimization result. NumPy array with flags NPY_ARRAY_CARRAY, type\n"
  "    NPY_DOUBLE, shape (n_features,), or at least an object convertable to\n"
  "    such a particular type of NumPy array.\n"
  "success : bool\n"
  "    True if optimization completed successfully, False otherwise.\n"
  "status : int\n"
  "    Numerical exit code indicating exit status. Typically 0 for normal\n"
  "    exit, positive int values for errors. Must not exceed INT_MAX.\n"
  "message : str\n"
  "    Message describing the optimizer cause of termination.\n"
  "fun_x : float\n"
  "    Final value of the objective function, fun in the OptimizeResult.\n"
  "n_fev : int\n"
  "    Number of objective evaluations, nfev in the OptimizeResult.\n"
  "n_iter : int\n"
  "    Number of solver iterations, nit in the OptimizeResult.\n"
  "jac_x : numpy.ndarray, default=None\n"
  "    Final gradient value. If provided, must be NumPy array with same\n"
  "    flags as x or a convertible object, shape (n_features,). Corresponds\n"
  "    to the jac attribute in the OptimizeResult.\n"
  "n_jev : int, default=None\n"
  "    Number of gradient evaluations, njev in the OptimizeResult.\n"
  "hess_x : numpy.ndarray, default=None\n"
  "    Final [approximate] Hessian value. If provided, must have same flags\n"
  "    as x or be a convertible object, shape (n_features, n_features).\n"
  "    Corresponds to the hess attribute in the OptimizeResult.\n"
  "n_hev : int, default=None\n"
  "    Number of Hessian evaluations, nhev in the OptimizeResult.\n"
  "hess_inv : numpy.ndarray, default=None\n"
  "    Inverse of the final [approximate] Hessian. If provided, must have\n"
  "    same flags as x or be a convertible object, shape\n"
  "    (n_features, n_features) like hess_x.\n"
  "maxcv : float, default=None\n"
  "    Maximum constraint violation."
  "\n\n"
  "Returns\n"
  "-------\n"
  "scipy.optimize.OptimizeResult"
);
// argument names known to populate_OptimizeResult
static const char *populate_OptimizeResult_argnames[] = {
  "x", "success", "status", "message", "fun_x", "n_fev", "n_iter",
  "jac_x", "n_jev", "hess_x", "n_hev", "hess_inv", "maxcv", NULL
};
/**
 * Python-accessible wrapper for internal function `populate_OptimizeResult`.
 * 
 * @param self `PyObject *` module (unused)
 * @param args `PyObject *` positional args tuple
 * @param kwargs `PyObject *` dict of keyword args, possibly `NULL`
 * @returns New `PyObject *` reference to a `scipy.optimize.OptimizeResult`
 *     populated with the specified arguments, `NULL` with exception on error.
 */
static PyObject *
populate_OptimizeResult(PyObject *self, PyObject *args, PyObject *kwargs)
{
  // optimization result, final gradient, final Hessian + inverse
  PyArrayObject *x, *jac_x, *hess_x, *hess_inv;
  x = jac_x = hess_x = hess_inv = NULL;
  // function value, maximum constraint violation
  PyObject *fun_x, *maxcv;
  fun_x = maxcv = NULL;
  // success, status, message
  int success, status;
  const char *message;
  // number of function, gradient, Hessian, evaluations, solver iterations.
  // note that n_jev, n_hev are set to -1 by default
  Py_ssize_t n_fev, n_jev, n_hev, n_iter;
  n_jev = n_hev = -1;
  // parse arguments. we convert PyArrayObject * later.
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "OpisO!nn|OnOnOO!",
      (char **) populate_OptimizeResult_argnames,
      &x, &success, &status, &message, &PyFloat_Type, &fun_x, &n_fev, &n_iter,
      &jac_x, &n_jev, &hess_x, &n_hev, &hess_inv, &PyFloat_Type, &maxcv
    )
  ) {
    return NULL;
  }
  // convert x to ndarray, NPY_DOUBLE with NPY_ARRAY_CARRAY flags. can ignore
  // the previous reference, which is borrowed.
  x = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) x, NPY_DOUBLE, NPY_ARRAY_CARRAY
  );
  if (x == NULL) {
    return NULL;
  }
  // x must not be empty, have only 1 dimension
  if (PyArray_SIZE(x) == 0) {
    PyErr_SetString(PyExc_ValueError, "x must be nonempty");
    goto except_x;
  }
  if (PyArray_NDIM(x) != 1) {
    PyErr_SetString(PyExc_ValueError, "x must be 1D");
    goto except_x;
  }
  // get n_features from x
  npy_intp n_features = PyArray_DIM(x, 0);
  // convert jac_x to ndarray if not NULL, same flags as x
  if (jac_x != NULL) {
    jac_x = (PyArrayObject *) PyArray_FROM_OTF(
      (PyObject *) jac_x, NPY_DOUBLE, NPY_ARRAY_CARRAY
    );
    if (jac_x == NULL) {
      goto except_x;
    }
    // must have same shape as x
    if (!PyArray_SAMESHAPE(x, jac_x)) {
      PyErr_SetString(PyExc_ValueError, "jac_x must have shape (n_features,)");
      goto except_jac_x;
    }
  }
  // convert hess_x to ndarray if not NULL, same flags as x
  if (hess_x != NULL) {
    hess_x = (PyArrayObject *) PyArray_FROM_OTF(
      (PyObject *) hess_x, NPY_DOUBLE, NPY_ARRAY_CARRAY
    );
    if (hess_x == NULL) {
      goto except_jac_x;
    }
    // must have 2 dimensions and have n_features in each dimension
    if (
      PyArray_NDIM(hess_x) != 2 ||
      PyArray_DIM(hess_x, 0) != n_features ||
      PyArray_DIM(hess_x, 1) != n_features
    ) {
      PyErr_SetString(
        PyExc_ValueError, "hess_x must have shape (n_features, n_features)"
      );
      goto except_hess_x;
    }
  }
  // convert hess_inv not ndarray if not NULL, same flags as x
  if (hess_inv != NULL) {
    hess_inv = (PyArrayObject *) PyArray_FROM_OTF(
      (PyObject *) hess_inv, NPY_DOUBLE, NPY_ARRAY_CARRAY
    );
    if (hess_inv == NULL) {
      goto except_hess_x;
    }
    // must have 2 dimensions and n_features in each dimension
    if (
      PyArray_NDIM(hess_inv) != 2 ||
      PyArray_DIM(hess_inv, 0) != n_features ||
      PyArray_DIM(hess_inv, 1) != n_features
    ) {
      PyErr_SetString(
        PyExc_ValueError, "hess_inv must have shape (n_features, n_features)"
      );
      goto except_hess_inv;
    }
  }
  // done converting, so feed to Py__mnewton_populate_OptimizeResult
  PyObject *res = Py__mnewton_populate_OptimizeResult(
    x, success, status, message, fun_x, jac_x, hess_x, hess_inv,
    n_fev, n_jev, n_hev, n_iter, maxcv
  );
  if (res == NULL) {
    goto except_hess_inv;
  }
  // clean up the new PyArrayObject * references (may be NULL) and return res
  Py_XDECREF(hess_inv);
  Py_XDECREF(hess_x);
  Py_XDECREF(jac_x);
  Py_DECREF(x);
  return res;
// clean up on exceptions
except_hess_inv:
  Py_XDECREF(hess_inv);
except_hess_x:
  Py_XDECREF(hess_x);
except_jac_x:
  Py_XDECREF(jac_x);
except_x:
  Py_DECREF(x);
  return NULL;
}

// docstring for lower_packed_copy wrapper
PyDoc_STRVAR(
  lower_packed_copy_doc,
  "lower_packed_copy(mat)"
  "\n--\n\n"
  "Python-accessible wrapper for internal function `lower_packed_copy`."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "mat : numpy.ndarray\n"
  "    Symmetric (not checked) matrix shape (n, n). Must be convertible to\n"
  "    type NPY_DOUBLE with flags NPY_ARRAY_IN_ARRAY, i.e. C-contiguous and\n"
  "    behaved, although not necessarily writable."
  "\n\n"
  "Returns\n"
  "-------\n"
  "numpy.ndarray\n"
  "    Lower triangle of mat stored as a 1D array, type NPY_DOUBLE."
);
/**
 * Python-accessible wrapper for `lower_packed_copy`.
 * 
 * @param self `PyObject *` module (unused)
 * @param arg `PyObject *` single positional arg
 * @returns New reference to `PyArrayObject *` on success, `NULL` on failure.
 */
static PyObject *
lower_packed_copy(PyObject *self, PyObject *arg)
{
  // original symmetric matrix and its packed lower triangle
  PyArrayObject *mat, *lower;
  // number of rows/columns
  npy_intp n_features;
  // pointers to data of mat and lower
  double *mat_data, *lower_data;
  // attempt to convert arg, the original matrix object (may not be ndarray),
  // into NPY_DOUBLE type and NPY_ARRAY_IN_ARRAY flags
  mat = (PyArrayObject *) PyArray_FROM_OTF(arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (mat == NULL) {
    return NULL;
  }
  // ensure that mat is nonempty and 2D, + get number of rows
  if (PyArray_SIZE(mat) == 0) {
    PyErr_SetString(PyExc_ValueError, "mat must be nonempty");
    goto except_mat;
  }
  if (PyArray_NDIM(mat) != 2) {
    PyErr_SetString(PyExc_ValueError, "mat must be 2D");
    goto except_mat;
  }
  n_features = PyArray_DIM(mat, 0);
  // check that mat is square (symmetry not checked)
  if (PyArray_DIM(mat, 1) != n_features) {
    PyErr_SetString(PyExc_ValueError, "mat must have shape (n, n)");
    goto except_mat;
  }
  // allocate new ndarray for lower, type NPY_DOUBLE, flags NPY_ARRAY_CARRAY.
  // lower only needs n_features * (n_features + 1) / 2 elements.
  npy_intp lower_dims[] = {n_features * (n_features + 1) / 2};
  lower = (PyArrayObject *) PyArray_SimpleNew(1, lower_dims, NPY_DOUBLE);
  if (lower == NULL) {
    goto except_mat;
  }
  // get data pointers and call Py__mnewton_lower_packed_copy
  mat_data = (double *) PyArray_DATA(mat);
  lower_data = (double *) PyArray_DATA(lower);
  Py__mnewton_lower_packed_copy(mat_data, lower_data, n_features);
  // clean up unneeded mat and return
  Py_DECREF(mat);
  return (PyObject *) lower;
// clean up on errors
except_mat:
  Py_DECREF(mat);
  return NULL;
}

// docstring for compute_mnewton_descent wrapper
PyDoc_STRVAR(
  compute_mnewton_descent_doc,
  "compute_mnewton_descent(hess, jac, beta=1e-3, tau_factor=2.)"
  "\n--\n\n"
  "Python-accessible wrapper for internal function `compute_mnewton_descent`."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "hess : numpy.ndarray\n"
  "    Current Hessian matrix of the objective, shape (n, n). Must be\n"
  "    convertible to type NPY_DOUBLE with flags NPY_ARRAY_IN_ARRAY, i.e.\n"
  "    C-contiguous and behaved, although not necessarily writable.\n"
  "jac : numpy.ndarray\n"
  "    Curretn gradient of the objective, shape (n,). Must be convertible\n"
  "    to type NPY_DOUBLE with flags NPY_ARRAY_IN_ARRAY.\n"
  "beta : float, default=1e-3\n"
  "    Minimum value to add to the diagonal of hess if a copy of hess must\n"
  "    be modified to be positive definite. Technically, only the lower\n"
  "    triangle of hess will be stored in packed format, and that same\n"
  "    memory will also hold the lower Choleksy factor of hess later on.\n"
  "tau_factor : float, default=2.\n"
  "    Value to scale the identity matrix added to hess at each iteration if\n"
  "    hess must be modified to be positive definite. Must be at least 1 in\n"
  "    theory but in practice, usually values >=2 are desired."
  "\n\n"
  "Returns\n"
  "-------\n"
  "numpy.ndarray\n"
  "    The [modified] Newton descent direction, shape (n,), type NPY_DOUBLE\n"
  "    with flags NPY_ARRAY_CARRAY (NPY_ARRAY_DEFAULT)."
);
// argument names known to compute_mnewton_descent
static const char *compute_mnewton_descent_argnames[] = {
  "fun", "jac", "x", "args", NULL
};
/**
 * Python-accessible wrapper for `compute_mnewton_descent`.
 * 
 * @param self `PyObject *` module (unused)
 * @param args `PyObject *` tuple of positional args
 * @param kwargs `PyObject *` dict of keyword arguments, may be `NULL`
 * @returns New reference to `PyArrayObject *` on success, `NULL` on failure.
 */
static PyObject *
compute_mnewton_descent(PyObject *self, PyObject *args, PyObject *kwargs)
{
  // Hessian matrix, gradient value, descent direction, beta, tau_factor
  PyArrayObject *hess, *jac, *dvec;
  double beta, tau_factor;
  // number of features, i.e. number of optimization variables
  npy_intp n_features;
  // parse arguments (all refs borrowed, so no cleanup)
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "OO|dd", (char **) compute_mnewton_descent_argnames,
      &hess, &jac, &beta, &tau_factor
    )
  ) {
    return NULL;
  }
  // convert hess, jac to PyArrayObject * NPY_DOUBLE, NPY_ARRAY_IN_ARRAY. it's
  // ok to discard original ref (address) since it is borrowed.
  hess = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) hess, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  if (hess == NULL) {
    return NULL;
  }
  jac = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) jac, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  if (jac == NULL) {
    goto except_hess;
  }
  // check that hess, jac are not empty
  if (PyArray_SIZE(hess) == 0) {
    PyErr_SetString(PyExc_ValueError, "hess must be nonempty");
    goto except_jac;
  }
  if (PyArray_SIZE(jac) == 0) {
    PyErr_SetString(PyExc_ValueError, "jac must be nonempty");
    goto except_jac;
  }
  // check shape of hess, which must be (n_features, n_features)
  if (PyArray_NDIM(hess) != 2) {
    PyErr_SetString(PyExc_ValueError, "hess must be 2D");
    goto except_jac;
  }
  if (PyArray_DIM(hess, 0) != PyArray_DIM(hess, 1)) {
    PyErr_SetString(PyExc_ValueError, "hess must have shape (n, n)");
    goto except_jac;
  }
  // get n_features using hess
  n_features = PyArray_DIM(hess, 0);
  // check shape of jac, which must bave (n_features,)
  if (PyArray_NDIM(jac) != 1) {
    PyErr_SetString(PyExc_ValueError, "jac must be 1D");
    goto except_jac;
  }
  if (PyArray_DIM(jac, 0) != n_features) {
    PyErr_SetString(PyExc_ValueError, "jac must have shape (n,)");
    goto except_jac;
  }
  // beta must be positive and tau_factor > 1
  if (beta <= 0) {
    PyErr_SetString(PyExc_ValueError, "beta must be positive");
    goto except_jac;
  }
  if (tau_factor <= 1) {
    PyErr_SetString(PyExc_ValueError, "tau_factor must be >1");
    goto except_jac;
  }
  // call Py__mnewton_compute_mnewton_descent and get result in dvec
  dvec = Py__mnewton_compute_mnewton_descent(hess, jac, beta, tau_factor);
  if (dvec == NULL) {
    goto except_jac;
  }
  // clean up and return
  Py_DECREF(jac);
  Py_DECREF(hess);
  return (PyObject *) dvec;
// clean up on error
except_jac:
  Py_DECREF(jac);
except_hess:
  Py_DECREF(hess);
  return NULL;
}

// _mnewton_internal methods (wrap internal functions in _mnewton)
static PyMethodDef _mnewton_internal_methods[] = {
  {
    "remove_specified_kwargs",
    (PyCFunction) remove_specified_kwargs,
    METH_VARARGS | METH_KEYWORDS, remove_specified_kwargs_doc
  },
  {
    "remove_unspecified_kwargs",
    (PyCFunction) remove_unspecified_kwargs,
    METH_VARARGS | METH_KEYWORDS, remove_unspecified_kwargs_doc
  },
  {
    "npy_frob_norm",
    (PyCFunction) npy_frob_norm,
    METH_O, npy_frob_norm_doc
  },
  {
    "tuple_prepend_single",
    (PyCFunction) tuple_prepend_single,
    METH_VARARGS | METH_KEYWORDS, tuple_prepend_single_doc
  },
  {
    "loss_only_fun_call",
    (PyCFunction) loss_only_fun_call,
    METH_VARARGS | METH_KEYWORDS, loss_only_fun_call_doc
  },
  {
    "compute_loss_grad",
    (PyCFunction) compute_loss_grad,
    METH_VARARGS | METH_KEYWORDS, compute_loss_grad_doc
  },
  {
    "compute_hessian",
    (PyCFunction) compute_hessian,
    METH_VARARGS | METH_KEYWORDS, compute_hessian_doc
  },
  {
    "populate_OptimizeResult",
    (PyCFunction) populate_OptimizeResult,
    METH_VARARGS | METH_KEYWORDS, populate_OptimizeResult_doc
  },
  {
    "lower_packed_copy",
    (PyCFunction) lower_packed_copy,
    METH_O, lower_packed_copy_doc
  },
  {
    "compute_mnewton_descent",
    (PyCFunction) compute_mnewton_descent,
    METH_VARARGS | METH_KEYWORDS, compute_mnewton_descent_doc
  },
  // sentinel marking end of array
  {NULL, NULL, 0, NULL}
};

// _mnewton_internal module definition
static PyModuleDef _mnewton_internal_module = {
  PyModuleDef_HEAD_INIT,
  // name, docstring, size = -1 to disable subinterpreter support, methods
  .m_name = "_mnewton_internal",
  .m_doc = "Wrappers for unit testing internal C functions in _mnewton.",
  .m_size = -1,
  .m_methods = _mnewton_internal_methods
};

// module initialization function
PyMODINIT_FUNC
PyInit__mnewton_internal(void)
{
  // import NumPy Array C API. automatically returns NULL on error.
  import_array();
  // import _mnewton C API.  automatically returns NULL on error.
  import__mnewton();
  // create module and return. NULL on error
  return PyModule_Create(&_mnewton_internal_module);
}