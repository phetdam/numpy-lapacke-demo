/**
 * @file _mnewton.c
 * @author Derek Huang <djh458@stern.nyu.edu>
 * @brief C implementation of Newton's method with Hessian modification.
 * 
 * The modified Hessian is the original Hessian with an added multiple of the
 * identity, checked to be positive definite by Cholesky decomposition. Method
 * is compatible as a frontend for scipy.optimize.minimize.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <limits.h>
#include <math.h>
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

/**
 * Remove a select subset of string keys from a kwargs dict.
 * 
 * Do NOT call without proper input checking. All keys must be string keys.
 * 
 * @param kwargs `PyObject *` implementing `PyDictObject` interface. All keys
 *     are strings, ex. the `kwargs` given to `PyArg_ParseTupleAndKeywords`.
 * @param droplist `char **` to `NULL`-terminated array of strings giving the
 *     names of string keys to drop from `kwargs`.
 * @param warn `int`, `!=0` to emit warning if a kwarg name in `droplist` is
 *     not in `kwargs`, `0` for silent no-op if kwarg name not in `kwargs`.
 * @returns The nonnegative number of names dropped from `kwargs` on success
 *     or `-1` on error with appropriate exception set.
 */
static Py_ssize_t
remove_specified_kwargs(PyObject *kwargs, const char **droplist, int warn)
{
  // number of names in kwargs dropped
  Py_ssize_t drops = 0;
  // index into droplist
  Py_ssize_t i = 0;
  // pointer to python string created from each key in droplist
  PyObject *key;
  // until we reach the end of droplist
  while (droplist[i] != NULL) {
    // create python string from droplist[i]. NULL on error
    key = PyUnicode_FromString(droplist[i]);
    if (key == NULL) {
      return -1;
    }
    // check that kwargs contains key. -1 on error, 0 for no, 1 for yes.
    int contains = PyDict_Contains(kwargs, key);
    // attempt to remove if key is in kwargs
    if (contains == 1) {
      // -1 on error, in which case we have to clean up
      if (PyDict_DelItem(kwargs, key) < 0) {
        goto except_key;
      }
      // on success, increment drops
      drops++;
    }
    // clean up on error
    else if (contains == -1) {
      goto except_key;
    }
    // else key not in kwargs. if warn, emit warning, else fall through
    else if (warn) {
      // returns -1 if exception is raised on warning, so we have to clean up
      if (
        PyErr_WarnFormat(
          PyExc_UserWarning, 1, "key %s not in kwargs", droplist[i]
        ) < 0
      ) {
        goto except_key;
      }
    }
    // clean up key and move on
    Py_DECREF(key);
    i++;
  }
  // everything cleaned up and we can return drops
  return drops;
// clean up before returning on error
except_key:
  Py_DECREF(key);
  return -1;
}

/**
 * Remove all keys from a kwargs dict except for a select subset of names.
 * 
 * Do NOT call without proper input checking. Naive two-loop implementation.
 * All the keys of the dict are expected to be strings.
 * 
 * @param kwargs `PyObject *` implementing `PyDictObject` interface. All keys
 *     are strings, ex. the `kwargs` given to `PyArg_ParseTupleAndKeywords`.
 * @param keeplist `char **` to `NULL`-terminated array of strings giving the
 *     names of string keys to keep in `kwargs`.
 * @param warn `int`, `!=0` to emit warning whenever an unspecified kwarg is
 *     dropped, `0` for silent dropping of unwanted kwargs.
 * @returns The nonnegative number of names dropped from `kwargs` on success
 *     or `-1` on error with appropriate exception set.
 */
static Py_ssize_t
remove_unspecified_kwargs(PyObject *kwargs, const char **keeplist, int warn)
{
  // number of names in kwargs dropped
  Py_ssize_t drops = 0;
  // get list of keys from kwargs
  PyObject *kwargs_keys = PyDict_Keys(kwargs);
  if (kwargs_keys == NULL) {
    return -1;
  }
  // get length of kwargs_keys. no need for error checking (will be a list)
  Py_ssize_t n_keys = PyList_GET_SIZE(kwargs_keys);
  /**
   * PyObject * to hold references to current key to check if in keeplist and a
   * key from keeplist to check against. function scope eases cleanup.
   */
  PyObject *kwargs_key_i, *keep_key_j;
  // for each key in kwargs_keys
  for (Py_ssize_t i = 0; i < n_keys; i++) {
    // get reference to the key (no checking). borrowed, so no Py_DECREF
    kwargs_key_i = PyList_GET_ITEM(kwargs_keys, i);
    // for each name in keeplist (indexed by j)
    Py_ssize_t j = 0;
    while (keeplist[j] != 0) {
      // create python string from keeplist[j]. NULL on error
      keep_key_j = PyUnicode_FromString(keeplist[j]);
      if (keep_key_j == NULL) {
        goto except;
      }
      // compare kwargs_key_i to keep_key_j. match is 1, 0 no match, -1 error.
      int key_match;
      key_match = PyObject_RichCompareBool(kwargs_key_i, keep_key_j, Py_EQ);
      // clean up keep_key_j with Py_DECREF (don't need it anymore)
      Py_DECREF(keep_key_j);
      // check value of key_match. if error, clean up
      if (key_match == -1) {
        goto except;
      }
      // if key_match is 1, kwargs_key_i in keeplist. break
      if (key_match == 1) {
        break;
      }
      // else move to next item in keeplist to compare against
      j++;
    }
    // if keeplist[j] == NULL, then no match. we drop kwargs_key_i
    if (keeplist[j] == NULL) {
      // -1 on failure, in which case we need to clean up
      if (PyDict_DelItem(kwargs, kwargs_key_i) < 0) {
        goto except;
      }
      // if warn is true, i.e != 0, emit UserWarning after drop
      if (warn) {
        // get name of key as string (don't need to deallocate). NULL on error
        const char *kwargs_str_i = PyUnicode_AsUTF8(kwargs_key_i);
        if (kwargs_str_i == NULL) {
          goto except;
        }
        // if warning issuance returns -1, exception was raised, so clean up
        if (
          PyErr_WarnFormat(
            PyExc_UserWarning, 1, "named argument %s removed from kwargs",
            kwargs_str_i
          ) < 0
        ) {
          goto except;
        }
      }
      // on success, increment drops
      drops++;
    }
    // don't Py_DECREF kwargs_key_i since it is borrowed!
  }
  // after dropping keys not in keeplist, clean up + return drops
  Py_DECREF(kwargs_keys);
  return drops;
// clean up before returning error
except:
  Py_DECREF(kwargs_keys);
  return -1;
}

/**
 * wrapper code for remove_specified_kwargs, remove_unspecified_kwargs that
 * lets us test these functions from Python. __INTELLISENSE__ always defined in
 * VS Code, so the defined(__INTELLISENSE__) lets Intellisense work here.
 */
#if defined(__INTELLISENSE__) || defined(EXPOSE_INTERNAL)
// flags to pass to remove_kwargs_dispatcher controlling whether to call
// remove_specified_kwargs or remove_unspecified_kwargs
#define REMOVE_KWARGS_SPECIFIED 0
#define REMOVE_KWARGS_UNSPECIFIED 1 
// argument names known to remove_kwargs_dispatcher
static const char *remove_kwargs_dispatcher_argnames[] = {
  "kwargs", "kwargs_list", "warn"
};
/**
 * Internal `remove_specified_kwargs`, `remove_unspecified_kwargs` dispatcher.
 * 
 * Since both functions have the same signatures and require the same input
 * checking, their respective `EXPOSED_*` methods can just wrap this function.
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
   * pass kwdict, kwargs_list, warn to remove_specified_kwargs or
   * remove_unspecified_kwargs depending on dispatch_flag and save the number
   * of keys dropped. drops will be -1 on error.
   */
  Py_ssize_t drops;
  if (dispatch_flag == REMOVE_KWARGS_SPECIFIED) {
    drops = remove_specified_kwargs(kwdict, kwargs_list, warn);
  }
  else {
    drops = remove_unspecified_kwargs(kwdict, kwargs_list, warn);
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

// docstring for EXPOSED_remove_specified_kwargs
PyDoc_STRVAR(
  EXPOSED_remove_specified_kwargs_doc,
  "EXPOSED_remove_specified_kwargs(kwargs, droplist, warn=True)"
  "\n--\n\n"
  EXPOSE_INTERNAL_NOTICE
  "\n\n"
  "Python-accessible wrapper for internal function ``remove_specified_kwargs``."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "kwargs : dict\n"
  "    :class:`dict` containing :class:`str` keys only, representing the\n"
  "    kwargs dict often unpacked to provide named arguments to functions.\n"
  "droplist : list\n"
  "    List of strings indicating which names in ``kwargs`` to drop.\n"
  "warn : bool, default=True\n"
  "    ``True`` to warn if a name in ``droplist`` is not in ``kwargs``,\n"
  "    ``False`` to not warn if a name in ``droplist`` is not in ``kwargs``."
  "\n\n"
  "Returns\n"
  "-------\n"
  "int\n"
  "    The number of names in ``droplist`` dropped from ``kwargs``."
);
/**
 * Python-accessible wrapper for `remove_specified_kwargs`.
 * 
 * @param self `PyObject *` module (unused)
 * @param args `PyObject *` tuple of positional args
 * @param kwargs `PyObject *` giving any keyword arguments, may be `NULL`
 * @returns New reference to `Py_None` on success, `NULL` on failure.
 */
static PyObject *
EXPOSED_remove_specified_kwargs(
  PyObject *self,
  PyObject *args, PyObject *kwargs
)
{
  // simply wrap remove_kwargs_dispatcher
  return remove_kwargs_dispatcher(args, kwargs, REMOVE_KWARGS_SPECIFIED);
}

// docstring for EXPOSED_remove_unspecified_kwargs
PyDoc_STRVAR(
  EXPOSED_remove_unspecified_kwargs_doc,
  "EXPOSED_remove_unspecified_kwargs(kwargs, droplist, warn=True)"
  "\n--\n\n"
  EXPOSE_INTERNAL_NOTICE
  "\n\n"
  "Python-accessible wrapper for internal ``remove_unspecified_kwargs``."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "kwargs : dict\n"
  "    :class:`dict` containing :class:`str` keys only, representing the\n"
  "    kwargs dict often unpacked to provide named arguments to functions.\n"
  "keeplist : list\n"
  "    List of strings indicating which names in ``kwargs`` to keep.\n"
  "warn : bool, default=True\n"
  "    ``True`` to warn if a name not ``keeplist`` has been removed from\n"
  "    ``kwargs``, ``False`` to otherwise not warn."
  "\n\n"
  "Returns\n"
  "-------\n"
  "int\n"
  "    The number of names in dropped from ``kwargs`` not in ``keeplist``."
);
/**
 * Python-accessible wrapper for `remove_specified_kwargs`.
 * 
 * @param self `PyObject *` module (unused)
 * @param args `PyObject *` tuple of positional args
 * @param kwargs `PyObject *` giving any keyword arguments, may be `NULL`
 * @returns New reference to `Py_None` on success, `NULL` on failure.
 */
static PyObject *
EXPOSED_remove_unspecified_kwargs(
  PyObject *self,
  PyObject *args, PyObject *kwargs
)
{
  // simply wrap remove_kwargs_dispatcher
  return remove_kwargs_dispatcher(args, kwargs, REMOVE_KWARGS_UNSPECIFIED);
}
#endif /* defined(__INTELLISENSE__) || defined(EXPOSE_INTERNAL) */

/**
 * Computes the Frobenius norm of a NumPy array.
 * 
 * Input array must be `NPY_DOUBLE`, `NPY_ARRAY_ALIGNED` flags. Can be empty.
 * 
 * @param ar `PyArrayObject *` ndarray. Must have required flags.
 * @returns `double` Frobenius norm of the NumPy array
 */
static double
npy_frob_norm(PyArrayObject *ar)
{
  // get size of array. if empty, return 0
  npy_intp ar_size = PyArray_SIZE(ar);
  // if empty, return 0
  if (ar_size == 0) {
    return 0;
  }
  // else get data pointer and compute norm
  double *ar_data = (double *) PyArray_DATA(ar);
  double ar_norm = 0;
  for (npy_intp i = 0; i < ar_size; i++) {
    ar_norm += ar_data[i] * ar_data[i];
  }
  return sqrt(ar_norm);
}

// wrapper code for npy_frob_norm to test from Python. __INTELLISENSE__ always
// defined in VS Code so defined(__INTELLISENSE__) lets Intellisense work here.
#if defined(__INTELLISENSE__) || defined(EXPOSE_INTERNAL)
// docstring for EXPOSED_npy_frob_norm
PyDoc_STRVAR(
  EXPOSED_npy_frob_norm_doc,
  "EXPOSED_npy_frob_norm(ar)"
  "\n--\n\n"
  "Python-accessible wrapper for internal functon ``npy_frob_norm``."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "ar : numpy.ndarray\n"
  "    NumPy array with flags ``NPY_ARRAY_ALIGNED``, type ``NPY_DOUBLE``, or\n"
  "    at least any object that can be converted to such a NumPy array."
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
EXPOSED_npy_frob_norm(PyObject *self, PyObject *arg)
{
  // convert arg to ndarray with NPY_DOUBLE type and NPY_ARRAY_ALIGNED flags
  PyArrayObject *ar;
  ar = (PyArrayObject *) PyArray_FROM_OTF(arg, NPY_DOUBLE, NPY_ARRAY_ALIGNED);
  if (ar == NULL) {
    return NULL;
  }
  // return PyFloatObject * from npy_frob_norm, NULL on error
  return (PyObject *) PyFloat_FromDouble(npy_frob_norm(ar));
}
#endif /* defined(__INTELLISENSE__) || defined(EXPOSE_INTERNAL) */

/**
 * Utility function for computing objective and gradient values with args.
 * 
 * The return values of the objective `fun` and gradient function `jac` are
 * checked to ensure that they are appropriate, i.e. that `fun` returns a
 * `PyFloatObject *` and that `jac` (if callable) has output that can be
 * converted to a `NPY_DOUBLE` ndarray, flags `NPY_ARRAY_IN_ARRAY`, with shape
 * `(n_features,)` to match the shape of the current parameter guess.
 * 
 * @param fun `PyObject *` to a callable, the objective function. Should have
 *     signature `fun(x, *args[1:])` and return a scalar.
 * @param jac `PyObject *` to a callable, the gradient function. Should have
 *     signature `jac(x, *args[1:])` and return a flat vector `(n_features,)`.
 *     May also be `True`, in which case `fun` is expected to return
 *     `(loss, grad)`, `loss` a scalar, `grad` shape `(n_features,)`.
 * @param args `PyTupleObject *` giving the arguments to pass to `fun` and
 *     `jac` (if callable), where the first argument must point to the current
 *     parameter guess, shape `(n_features,)`.
 * @returns New `PyTupleObject *` where the first element points to the
 *     objective value as a `PyFloatObject *` and the second element points to
 *     the gradient value as a `PyArrayObject *` shape `(n_features,)`. On
 *     error, `NULL` is returned, and an exception set.
 */
static PyTupleObject *
compute_loss_grad(PyObject *fun, PyObject *jac, PyTupleObject *args)
{
  // function value and temp variables for the return values of fun, jac
  PyObject *fun_x, *temp_f, *temp_g;
  // gradient value
  PyArrayObject *jac_x;
  // compute initial loss and gradient. if jac is Py_True, call fun instead.
  if (jac == Py_True) {
    temp_f = PyObject_CallObject(fun, (PyObject*) args);
    // clean up on error, i.e. temp_f == NULL
    if (temp_f == NULL) {
      return NULL;
    }
    // try to get loss from first index of temp_f, which may not be a tuple.
    // Py_XINCREF it so on failure, no-op, while on success, we own a ref.
    fun_x = PyTuple_GetItem(temp_f, 0);
    Py_XINCREF(fun_x);
    /**
     * try to get gradient from second index of temp_f (tuple). Py_XINCREF it
     * so on failure, no-op, while on success, we own a reference. don't need
     * temp_f any more, so just Py_DECREF it.
     */
    temp_g = PyTuple_GetItem(temp_f, 1);
    Py_XINCREF(temp_g);
    Py_DECREF(temp_f);
  }
  // else call fun and jac separately to compute fun_x, jac_x
  else {
    fun_x = PyObject_CallObject(fun, (PyObject *) args);
    temp_g = PyObject_CallObject(jac, (PyObject *) args);
  }
  // if fun_x NULL, error. temp_g may be NULL or a new reference so XDECREF it
  if (fun_x == NULL) {
    Py_XDECREF(temp_g);
    return NULL;
  }
  // if temp_g NULL, error. fun_x may be NULL or a new reference so XDECREF it
  if (temp_g == NULL) {
    Py_XDECREF(fun_x);
    return NULL;
  }
  // check that fun_x is a [subtype] of PyFloatObject. if not, error. must also
  // Py_DECREF temp_g, which is a new reference
  if (!PyFloat_Check(fun_x)) {
    PyErr_SetString(PyExc_TypeError, "fun must return a float");
    Py_DECREF(temp_g);
    goto except_fun_x;
  }
  // convert temp_g to ndarray with type NPY_DOUBLE, NPY_ARRAY_INARRAY (no
  // need to be writeable). may create data copy. Py_DECREF unneeded temp_g.
  jac_x = (PyArrayObject *) PyArray_FROM_OTF(
    temp_g, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  Py_DECREF(temp_g);
  // jac_x NULL if error as usual + clean up fun_x as well
  if (jac_x == NULL) {
    goto except_fun_x;
  }
  // else success. check that jac_x, x have same shape, where x is the first
  // element of args. if shape is different, error
  if (!PyArray_SAMESHAPE((PyArrayObject *) PyTuple_GET_ITEM(args, 0), jac_x)) {
    PyErr_SetString(PyExc_ValueError, "gradient must have shape (n_features,)");
    goto except_jac_x;
  }
  // create new tuple to hold fun_x, jac_x. no Py_DECREF as refs are stolen
  PyTupleObject *res = (PyTupleObject *) PyTuple_New(2);
  PyTuple_SET_ITEM(res, 0, fun_x);
  PyTuple_SET_ITEM(res, 1, (PyObject *) jac_x);
  return res;
// clean up when handling error
except_jac_x:
  Py_DECREF(jac_x);
except_fun_x:
  Py_DECREF(fun_x);
  return NULL;
}

/**
 * Utility function for computing Hessian with args.
 * 
 * The return values of the Hessian function `hess` is checked to ensure that
 * it is appropriate, i.e. that `hess` has output that can be converted to a
 * `NPY_DOUBLE` ndarray, flags `NPY_ARRAY_IN_ARRAY`, with shape
 * `(n_features, n_features)`. Current parameter has shape `(n_features,)`.
 * 
 * @param hess `PyObject *` to a callable, the Hessian function. Should have
 *     signature `hess(x, *args[1:])` and return output convertible to a
 *     ndarray shape `(n_features, n_features)`.
 * @param args `PyTupleObject *` giving the arguments to pass to `hess`, also
 *     passed to `fun`, `jac` in `compute_loss_grad`. The first argument must
 *     point to the current parameter guess, shape `(n_features,)`.
 * @returns New `PyArrayObject *` shape `(n_features, n_features)` giving the
 *     Hessian, type `NPY_DOUBLE`, flags `NPY_ARRAY_IN_ARRAY`. On error, `NULL`
 *     is returned, and an exception set.
 */
static PyArrayObject *
compute_hessian(PyObject *hess, PyTupleObject *args)
{
  // hessian value and temp to hold function return values
  PyArrayObject *hess_x;
  PyObject *temp_f;
  // call hess with the provided arguments. NULL on error
  temp_f = PyObject_CallObject(hess, (PyObject *) args);
  if (temp_f == NULL) {
    return NULL;
  }
  // else convert to NPY_DOUBLE, flags NPY_ARRAY_IN_ARRAY. Py_DECREF temp_f.
  hess_x = (PyArrayObject *) PyArray_FROM_OTF(
    temp_f, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  Py_DECREF(temp_f);
  // on error, hess_x NULL. nothing to clean so return NULL
  if (hess_x == NULL) {
    return NULL;
  }
  // get features from x, the first elements of args. note x ref is borrowed.
  PyArrayObject *x = (PyArrayObject *) PyTuple_GET_ITEM(args, 0);
  npy_intp n_features = PyArray_DIM(x, 0);
  // check that hess_x has only 2 dimensions, each same as size of x, and also
  // that hess_x has shape (n_features, n_features)
  if (
    PyArray_NDIM(hess_x) != 2 || PyArray_DIM(hess_x, 0) != n_features ||
    PyArray_DIM(hess_x, 1) != n_features
  ) {
    PyErr_SetString(
      PyExc_ValueError, "Hessian must have shape (n_features, n_features)"
    );
    Py_DECREF(hess_x);
    return NULL;
  }
  // if shape of Hessian is correct, return
  return hess_x;
}

// docstring for mnewton
PyDoc_STRVAR(
  mnewton_doc,
  "mnewton(fun, x0, *, args=(), jac=None, hess=None, gtol=1e-4,\n"
  "maxiter=1000, alpha=0.5, beta=1e-3, gamma=0.8, **ignored)"
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
  "Parameters\n"
  "----------\n"
  "fun : callable\n"
  "    Objective function to minimize, with signature ``fun(x, *args)`` and\n"
  "    returning a :class:`numpy.ndarray` with shape ``(n_features,)``.\n"
  "x0 : numpy.ndarray\n"
  "    Initial solution guess, shape ``(n_features,)``\n"
  "args : tuple, default=()\n"
  "    Positional arguments to pass to ``fun``, ``jac``, and ``hess``\n"
  "jac : callable, default=None\n"
  "    Gradient of ``fun``, with signature ``grad(x, *args)`` and returning\n"
  "    a :class:`numpy.ndarray` with shape ``(n_features,)``. If ``True``,\n"
  "    then ``fun`` is expected to return ``(loss, grad)``. If not\n"
  "    specified, a :class:`RuntimeError` will be raised.\n"
  "hess : callable, default=None\n"
  "    Hessian of ``fun``, with signature ``hess(x, *args)`` and returning\n"
  "    a :class:`numpy.ndarray` with shape ``(n_features, n_features)``. If\n"
  "    not specified, a :class:`RuntimeError` will be raised.\n"
  "\n\n"
  ".. [#] Nocedal, J., & Wright, S. (2006). *Numerical Optimization*.\n"
  "   Springer Science and Business Media."
);
// mnewton argument names. PyArg_ParseTupleAndKeywords requires ending NULL.
static const char *mnewton_argnames[] = {
  "fun", "x0", "args", "jac", "hess", "gtol",
  "maxiter", "alpha", "beta", "gamma", NULL
};
/**
 * mnewton arguments to be specifically dropped from kwargs before calling
 * PyArg_ParseTupleAndKeywords; these are scipy.optimize.minimize arguments.
 * see lines 596-602 in _minimize.py in scipy.optimize.
 */
static const char *mnewton_scipy_argnames[] = {
  "bounds", "constraints", "callback"
};
/**
 * Newton's method with Hessian modification.
 * 
 * Uses Cholesky decomposition to check if the Hessian plus the scaled identity
 * matrix is positive definite before computing the search direction.
 * 
 * @param self `PyObject *` module (unused)
 * @param args `PyObject *` positional args tuple
 * @param kwargs `PyObject *` keyword args dict
 * @returns `scipy.optimize.OptimizeResult` on success, `NULL` with exception
 *     set on error, whether it be Python or CBLAS/LAPACKE.
 */
static PyObject *
mnewton(PyObject *self, PyObject *args, PyObject *kwargs)
{
  // PyObject * for fun, jac, hess (must be callable), and function value.
  PyObject *fun, *jac, *hess, *fun_x;
  fun = jac = hess = fun_x = NULL;
  // PyArrayObject * for the current parameter guess, gradient, and
  // [approximate] Hessian values. will be returned in OptimizeResult.
  PyArrayObject *x, *jac_x, *hess_x;
  x = jac_x = hess_x = NULL;
  /**
   * function arguments to pass to fun, jac, hess; originally corresponds to
   * the args parameter in the Python signature but will be modified to point
   * to a copy equivalent to (x, *args).
   */
  PyTupleObject *fun_args = NULL;
  /**
   * number of solver iterations, number of objective evaluations, number of
   * gradient evaluations, number of hessian evaluations (includes number of
   * Cholesky decompositions performed during modificaton)
   */
  Py_ssize_t n_iter, n_fev, n_jev, n_hev;
  n_iter = n_fev = n_jev = n_hev = 0;
  // gradient tolerance, max iterations, alpha, beta, gamma
  double gtol = 1e-4;
  Py_ssize_t maxiter = 1000;
  double alpha = 0.5;
  double beta = 1e-3;
  double gamma = 0.8;
  /**
   * scipy.optimize.minimize requires that custom minimizers accept the
   * arguments fun, args (fun_args), jac, hess, hessp, bounds, constraints,
   * callback, as well as keyword arguments taken by the custom minimizer. this
   * function doesn't use bounds, constraints, callback, so we need to remove
   * them since only fun, x0 are allowed to be positional. any other unwanted
   * keyword arguments will also be dropped with warnings. technically mnewton
   * should not restrict the named arguments to be keyword-only, the way it
   * will be called allows us to do this. see lines 596-602 of
   * https://github.com/scipy/scipy/blob/master/scipy/optimize/_minimize.py
   */
  // get size of kwargs, i.e. number of key-value pairs. kwargs can be NULL.
  Py_ssize_t kwargs_size = (kwargs == NULL) ? 0 : PyDict_Size(kwargs);
  // we don't need to remove any names if size is 0
  if (kwargs_size > 0) {
    // remove bounds, constraints, callback from kwargs silently if present.
    // -1 on error, in which case we just return NULL (no new refs)
    if (remove_specified_kwargs(kwargs, mnewton_scipy_argnames, 0) < 0) {
      return NULL;
    }
    // remove any other unwanted kwargs if present but with warnings. -1 on
    // error, no new refs so again just return NULL
    if (remove_unspecified_kwargs(kwargs, mnewton_argnames, 1) < 0) {
      return NULL;
    }
  }
  // call PyArg_ParseTupleAndKeywords to parse arguments. no new refs yet.
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "OO|$O!OOdnddd", (char **) mnewton_argnames,
      &fun, &x, &PyTuple_Type, (PyObject **) &fun_args, &jac, &hess,
      &gtol, &maxiter, &alpha, &beta, &gamma
    )
  ) {
    return NULL;
  }
  // check arguments manually. first, fun must be callable
  if (!PyCallable_Check(fun)) {
    PyErr_SetString(PyExc_TypeError, "fun must be callable");
    return NULL;
  }
  // now we try to convert x to NPY_DOUBLE array with NPY_ARRAY_CARRAY flags.
  // the new array is enforced to be a copy of the original. NULL on error
  PyArrayObject *temp_ar = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) x, NPY_DOUBLE, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY
  );
  if (temp_ar == NULL) {
    return NULL;
  }
  // on success, overwrite x with temp_ar (original ref borrowed, no Py_DECREF)
  x = temp_ar;
  // x0 must not be empty and must have shape (n_features,)
  if (PyArray_SIZE(x) == 0) {
    PyErr_SetString(PyExc_ValueError, "x0 must be nonempty");
    goto except_x;
  }
  if (PyArray_NDIM(x) != 1) {
    PyErr_SetString(PyExc_ValueError, "x0 must have shape (n_features,)");
    goto except_x;
  }
  // jac must be either callable or Py_True
  if (!PyCallable_Check(jac) && jac != Py_True) {
    PyErr_SetString(PyExc_TypeError, "jac must be callable or True");
    goto except_x;
  }
  // hess needs to be callable
  if (!PyCallable_Check(hess)) {
    PyErr_SetString(PyExc_TypeError, "hess must be provided and be callable");
    goto except_x;
  }
  // gtol, maxiter, beta must be positive
  if (gtol <= 0) {
    PyErr_SetString(PyExc_ValueError, "gtol must be positive");
    goto except_x;
  }
  if (maxiter < 1) {
    PyErr_SetString(PyExc_ValueError, "maxiter must be positive");
    goto except_x;
  }
  if (beta <= 0) {
    PyErr_SetString(PyExc_ValueError, "beta must be positive");
    goto except_x;
  }
  // alpha and gamma must be in (0, 1)
  if (alpha <= 0 || alpha >= 1) {
    PyErr_SetString(PyExc_ValueError, "alpha must be in (0, 1)");
    goto except_x;
  }
  if (gamma <= 0 || gamma >= 1) {
    PyErr_SetString(PyExc_ValueError, "gamma must be in (0, 1)");
    goto except_x;
  }
  // create new tuple using contents of fun_args where the first argument is
  // the pointer to x. this makes calling fun, jac, hess much easier.
  PyTupleObject *_fun_args = fun_args;
  fun_args = (PyTupleObject *) PyTuple_New(1 + PyTuple_GET_SIZE(_fun_args));
  // no need to Py_DECREF _fun_args since it is borrowed ref
  if (fun_args == NULL) {
    goto except_x;
  }
  // Py_INCREF x and set first index of fun_args. note reference is stolen.
  Py_INCREF(x);
  PyTuple_SET_ITEM(fun_args, 0, (PyObject *) x);
  // fill fun_args with stolen Py_INCREF'd references from _fun_args
  for (Py_ssize_t i = 1; i < PyTuple_GET_SIZE(fun_args); i++) {
    PyObject *_fun_args_i = PyTuple_GET_ITEM(_fun_args, i - 1);
    Py_INCREF(_fun_args_i);
    PyTuple_SET_ITEM(fun_args, i, _fun_args_i);
  }
  // compute initial loss and gradient using compute_loss_grad. recall that the
  // first element of fun_args is a reference to x.
  PyTupleObject *temp_tp = compute_loss_grad(fun, jac, fun_args);
  // if temp_tp is NULL, clean up
  if (temp_tp == NULL) {
    goto except_fun_args;
  }
  // on success, assign temp_tp[0], temp_tp[1] to fun_x, jac_x + Py_INCREF
  fun_x = PyTuple_GET_ITEM(temp_tp, 0);
  jac_x = (PyArrayObject *) PyTuple_GET_ITEM(temp_tp, 1);
  Py_INCREF(fun_x);
  Py_INCREF(jac_x);
  // increment n_fev, n_jev to count objective + gradient evals
  n_fev++;
  n_jev++;
  // compute initial Hessian. NULL on error and clean up
  hess_x = compute_hessian(hess, fun_args);
  if (hess_x == NULL) {
    goto except_fun_jac_x;
  }
  // if successful, increment n_hev to count Hessian evals
  n_hev++;
  // optimize while not converged, i.e. avg. gradient norm less than tolerance
  while (npy_frob_norm(jac_x) < gtol) {

    // TODO: write actual optimization algorithm

    // done with fun_x, jac_x, hess_x, so we compute next values of fun_x,
    // jac_x, hess_x using updated x. can Py_DECREF all of these.
    Py_DECREF(fun_x);
    Py_DECREF(jac_x);
    Py_DECREF(hess_x);
    // compute next loss + gradient
    temp_tp = compute_loss_grad(fun, jac, fun_args);
    // clean up on error. note we already Py_DECREF fun_x, jac_x, hess_x
    if (temp_tp == NULL) {
      goto except_fun_args;
    }
    // else again assign temp_tp[0], temp_tp[1] to fun_x, jac_x + Py_INCREF
    fun_x = PyTuple_GET_ITEM(temp_tp, 0);
    jac_x = (PyArrayObject *) PyTuple_GET_ITEM(temp_tp, 1);
    Py_INCREF(fun_x);
    Py_INCREF(jac_x);
    // increment n_fev, n_jev to count objective + gradient evals
    n_fev++;
    n_jev++;
    // compute next hessian. NULL on error
    hess_x = compute_hessian(hess, fun_args);
    if (hess_x == NULL) {
      goto except_fun_jac_x;
    }
    // on success, increment n_hev, n_iter to count Hessian evals + iterations
    n_hev++;
    n_iter++;
  }

  // TODO: populate members of the OptimizeResult

  // since PyObject_SetAttrString internally Py_INCREFs, clean up and return
  Py_DECREF(hess_x);
  Py_DECREF(jac_x);
  Py_DECREF(fun_x);
  Py_DECREF(fun_args);
  //Py_DECREF(x);
  return (PyObject *) x;
// clean up on error
except_fun_jac_x:
  Py_DECREF(jac_x);
  Py_DECREF(fun_x);
except_fun_args:
  Py_DECREF(fun_args);
except_x:
  Py_DECREF(x);
  return NULL;
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
  {
    "EXPOSED_remove_specified_kwargs",
    (PyCFunction) EXPOSED_remove_specified_kwargs,
    METH_VARARGS | METH_KEYWORDS, EXPOSED_remove_specified_kwargs_doc
  },
  {
    "EXPOSED_remove_unspecified_kwargs",
    (PyCFunction) EXPOSED_remove_unspecified_kwargs,
    METH_VARARGS | METH_KEYWORDS, EXPOSED_remove_unspecified_kwargs_doc
  },
  {
    "EXPOSED_npy_frob_norm",
    (PyCFunction) EXPOSED_npy_frob_norm,
    METH_O, EXPOSED_npy_frob_norm_doc
  },
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