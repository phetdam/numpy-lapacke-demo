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

#define MNEWTON_MODULE
#include "mnewton.h"

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

/**
 * Builds new tuple from an object (first element) and another existing tuple.
 * 
 * If the object is `x` and the existing tuple is `tp`, then in Python the
 * operation would be the same as returning `(x, *tp)`.
 * 
 * @param x `PyObject *` object to add at the beginning of the new tuple.
 * @param old_tp `PyTupleObject *` old tuple whose elements are placed in order
 *     after `x` in the newly created tuple. `old` may be `NULL`, in which
 *     case a tuple containing only `x` is returned.
 * @returns New `PyTupleObject *` reference where the first element in `x` and
 *     the other elements are the elements of `old` in order, `NULL` on error.
 */
static PyTupleObject *
tuple_prepend_single(PyObject *x, PyTupleObject *old_tp)
{
  // new tuple to return, size of old_tp, temp to hold old_tp elements
  PyTupleObject *new_tp;
  Py_ssize_t old_size;
  PyObject *old_tp_i;
  // if old_tp is NULL, new tuple will only contain x
  if (old_tp == NULL) {
    new_tp = (PyTupleObject *) PyTuple_New(1);
  }
  // else will also contain the elements of old_tp
  else {
    old_size = PyTuple_GET_SIZE(old_tp);
    new_tp = (PyTupleObject *) PyTuple_New(1 + old_size);
  }
  // NULL on error
  if (new_tp == NULL) {
    return NULL;
  }
  // else populate first element with x (ref stolen)
  Py_INCREF(x);
  PyTuple_SET_ITEM(new_tp, 0, x);
  // if old_tp is not NULL, then also fill with its elements + return
  if (old_tp != NULL) {
    for (Py_ssize_t i = 0; i < old_size; i++) {
      // Py_INCREF ith element of old_tp, assign to (i + 1)th spot in new_tp
      old_tp_i = PyTuple_GET_ITEM(old_tp, i);
      Py_INCREF(old_tp_i);
      PyTuple_SET_ITEM(new_tp, i + 1, old_tp_i);
    }
  }
  return new_tp;
}

/**
 * Internal function for computing objective and gradient values with args.
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
 * Internal function for computing Hessian with args.
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

/**
 * Internal function for populating a `scipy.optimize.OptimizeResult`.
 * 
 * All `PyArrayObject *` must have type `NPY_DOUBLE`, `NPY_ARRAY_CARRAY` flags.
 * 
 * @param x `PyArrayObject *` optimization result, shape `(n_features,)`.
 * @param success `0` for failure, `!= 0` for optimization success.
 * @param status Termination status. Set to `0` for normal exit, `1` on error.
 * @param message `const char *` to message describing cause of termination.
 * @param fun_x `PyObject *` final objective value, either `PyFloatObject *`
 *     or a subclass that results in `PyFloat_Check` returning true.
 * @param jac_x `PyArrayObject *` final gradient, shape `(n_features,)`. Can be
 *     `NULL` to skip addition of the `jac` attribute to the `OptimizeResult`.
 * @param hess_x `PyArrayObject *` final [approximate] Hessian, shape
 *     `(n_features, n_features)`. Can be `NULL` to skip addition of the `hess`
 *     attribute to the `OptimizeResult`.
 * @param hess_inv `PyArrayObject *` final inverse [approximate] Hessian, shape
 *     `(n_features, n_features)`. Can be `NULL` to skip addition of the
 *     `hess_inv` attribute to the `OptimizeResult`.
 * @param n_fev `Py_ssize_t` number of objective function evaluations
 * @param n_jev `Py_ssize_t` number of gradient function evaluations. If `-1`,
 *     then the `njev` attribute will not be added to the `OptimizeResult`.
 * @param n_hev `Py_ssize_t` number of Hessian function evaluations. If `-1`,
 *     then the `nhev` attribute will not be added to the `OptimizeResult`.
 * @param n_iter `Py_ssize_t` number of solver iterations
 * @param maxcv `PyObject *` maximum constraint violation, either
 *     `PyFloatObject *` or a subclass that results in `PyFloat_Check`
 *     returning true. Can be `NULL` to skip addition of the `maxcv` attribute.
 * @returns New `OptimizeResult` reference with populated attributes, `NULL`
 *     on error with exception set as usual.
 */
static PyObject *
populate_OptimizeResult(
  PyArrayObject *x,
  int success, int status, const char *message,
  PyObject *fun_x,
  PyArrayObject *jac_x, PyArrayObject *hess_x, PyArrayObject *hess_inv,
  Py_ssize_t n_fev, Py_ssize_t n_jev, Py_ssize_t n_hev, Py_ssize_t n_iter,
  PyObject *maxcv
)
{
  // attempt to import scipy.optimize. NULL on error. if already imported, this
  // is just a new reference to the module in sys.modules.
  PyObject *opt_mod = PyImport_ImportModule("scipy.optimize");
  if (opt_mod == NULL) {
    return NULL;
  }
  // get the OptimizeResult class member from the module
  PyObject *opt_class = PyObject_GetAttrString(opt_mod, "OptimizeResult");
  if (opt_class == NULL) {
    goto except_opt_mod;
  }
  // call OptimizeResult with no args to get an empty OptimizeResult
  PyObject *opt_res = PyObject_CallObject(opt_class, NULL);
  if (opt_res == NULL) {
    goto except_opt_class;
  }
  // add x to instance. note Py_INCREF is done internally. -1 on error.
  if (PyObject_SetAttrString(opt_res, "x", (PyObject *) x) < 0) {
    goto except_opt_res;
  }
  // create pointer to Py_True or Py_False, depending on value of success
  PyObject *py_success = (success) ? Py_True : Py_False;
  Py_INCREF(py_success);
  // create PyLongObject * wrapper for status. NULL on error
  PyObject *py_status = PyLong_FromLong(status);
  if (py_status == NULL) {
    goto except_py_success;
  }
  // create PyUnicodeObject * from message
  PyObject *py_message = PyUnicode_FromString(message);
  if (py_message == NULL) {
    goto except_py_status;
  }
  // add py_success, py_status, py_message to instance. -1 on errors.
  if (PyObject_SetAttrString(opt_res, "success", py_success) < 0) {
    goto except_py_message;
  }
  if (PyObject_SetAttrString(opt_res, "status", py_status) < 0) {
    goto except_py_message;
  }
  if (PyObject_SetAttrString(opt_res, "message", py_message) < 0) {
    goto except_py_message;
  }
  // add fun_x, jac_x, hess_x, hess_inv to instance. jac_x, hess_x, hess_inv
  // are allowed to be NULL, in which case they are not added. -1 on errors.
  if (PyObject_SetAttrString(opt_res, "fun", fun_x) < 0) {
    goto except_py_message;
  }
  if (
    jac_x != NULL &&
    PyObject_SetAttrString(opt_res, "jac", (PyObject *) jac_x) < 0
  ) {
    goto except_py_message;
  }
  if (
    hess_x != NULL &&
    PyObject_SetAttrString(opt_res, "hess", (PyObject *) hess_x) < 0
  ) {
    goto except_py_message;
  }
  if (
    hess_inv != NULL &&
    PyObject_SetAttrString(opt_res, "hess_inv", (PyObject *) hess_inv) < 0
  ) {
    goto except_py_message;
  }
  // create PyLongObject * for n_fev, n_jev, n_hev, n_iter. note that py_njev,
  // py_nhev are set to NULL if -1 and will not be added.
  PyObject *py_nfev, *py_njev, *py_nhev, *py_nit;
  // number of function evaluations as PyLongObject *
  py_nfev = PyLong_FromSsize_t(n_fev);
  if (py_nfev == NULL) {
    goto except_py_message;
  }
  // n_jev == -1 => py_njev = NULL, otherwise get PyLongObject *
  if (n_jev == -1) {
    py_njev = NULL;
  }
  else {
    py_njev = PyLong_FromSsize_t(n_jev);
    // NULL on error
    if (py_njev == NULL) {
      goto except_py_nfev;
    }
  }
  // n_hev == -1 => py_nhev = NULL, otherwise get PyLongObject *
  if (n_hev == -1) {
    py_nhev = NULL;
  }
  else {
    py_nhev = PyLong_FromSsize_t(n_hev);
    // NULL on error. note that py_njev is Py_XDECREF'd in cleanup.
    if (py_nhev == NULL) {
      goto except_py_njev;
    }
  }
  // number of solver iterations as PyLongObject *
  py_nit = PyLong_FromSsize_t(n_iter);
  if (py_nit == NULL) {
    goto except_py_nhev;
  }
  // add py_nfev, py_njev, py_nhev, py_nit to instance. -1 on errors. again,
  // py_njev, py_nhev may be NULL, hence the extra checks.
  if (PyObject_SetAttrString(opt_res, "nfev", py_nfev) < 0) {
    goto except_py_nit;
  }
  if (py_njev != NULL && PyObject_SetAttrString(opt_res, "njev", py_njev) < 0) {
    goto except_py_nit;
  }
  if (py_nhev != NULL && PyObject_SetAttrString(opt_res, "nhev", py_nhev) < 0) {
    goto except_py_nit;
  }
  if (PyObject_SetAttrString(opt_res, "nit", py_nit) < 0) {
    goto except_py_nit;
  }
  // create maxcv attribute, where if NULL we skip and don't add. -1 on error
  if (maxcv != NULL && PyObject_SetAttrString(opt_res, "maxcv", maxcv) < 0) {
    goto except_py_nit;
  }
  // clean up any new references, as the underlying objects are Py_INCREF'd
  // internally by PyObject_SetAttrString, and then return opt_res
  Py_DECREF(py_nit);
  Py_XDECREF(py_nhev);
  Py_XDECREF(py_njev);
  Py_DECREF(py_nfev);
  Py_DECREF(py_message);
  Py_DECREF(py_status);
  Py_DECREF(py_success);
  Py_DECREF(opt_class);
  Py_DECREF(opt_mod);
  return opt_res;
// clean up on exceptions
except_py_nit:
  Py_DECREF(py_nit);
except_py_nhev:
  Py_XDECREF(py_nhev);
except_py_njev:
  Py_XDECREF(py_njev);
except_py_nfev:
  Py_DECREF(py_nfev);
except_py_message:
  Py_DECREF(py_message);
except_py_status:
  Py_DECREF(py_status);
except_py_success:
  Py_DECREF(py_success);
except_opt_res:
  Py_DECREF(opt_res);
except_opt_class:
  Py_DECREF(opt_class);
except_opt_mod:
  Py_DECREF(opt_mod);
  return NULL;
}

// Python-accessible wrapper for populate_OptimizeResult. __INTELLISENSE__ is
// defined in VS Code so defined(__INTELLISENSE__) lets Intellisense work.
#if defined(__INTELLISENSE__) || defined(EXPOSE_INTERNAL)

#endif /* defined(__INTELLISENSE__)  || defined(EXPOSE_INTERNAL) */

/**
 * Returns the Cholesky factor of a p.d. Hessian matrix, which may be modified.
 * 
 * Saves memory by modifying a copy of only the lower triangular portion of the
 * Hessian matrix provided to the function.
 * 
 * @param hess `PyArrayObject *` Hessian matrix, type `NPY_DOUBLE`, flags
 *     `NPY_ARRAY_CARRAY`, shape `(n_features, n_features)`. If not
 *     sufficiently positive definite, a multiple of the identity will be
 *     added to a copy of its lower triangle, which will be then be Cholesky
 *     factored when it is sufficiently positive definite.
 * @param beta `double` giving the minimum value to add to the diagonal of the
 *     copied lower triangular portion of `hess` during computation.
 * @returns New `PyArrayObject *` reference that holds the lower triangular
 *     Cholesky factor of `hess`, or `hess` plus a multiple of the identity
 *     which is sufficiently positive definite, in packed format. On error,
 *     `NULL` is returned with an exception set.
 */
static PyArrayObject *
cholesky_spd_hessian_update(PyArrayObject *hess, double beta)
{
  Py_RETURN_NONE;
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
  "bounds", "constraints", "callback", NULL
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
      &fun, &x, &PyTuple_Type, &fun_args, &jac, &hess,
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
  if (jac == NULL || (!PyCallable_Check(jac) && jac != Py_True)) {
    PyErr_SetString(PyExc_TypeError, "jac must be callable or True");
    goto except_x;
  }
  // hess needs to be callable
  if (hess == NULL || !PyCallable_Check(hess)) {
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
  /**
   * create new tuple using contents of fun_args where the first argument is
   * the pointer to x. this makes calling fun, jac, hess much easier. it's ok
   * to overwrite the ref since it is borrowed.
   */
  fun_args = tuple_prepend_single((PyObject *) x, fun_args);
  // compute initial loss and gradient using compute_loss_grad. recall that the
  // first element of fun_args is a reference to x.
  PyTupleObject *temp_tp = compute_loss_grad(fun, jac, fun_args);
  // if temp_tp is NULL, clean up
  if (temp_tp == NULL) {
    goto except_fun_args;
  }
  // on success, assign temp_tp[0], temp_tp[1] to fun_x, jac_x + Py_INCREF.
  // don't need temp_tp anymore, so Py_DECREF it
  fun_x = PyTuple_GET_ITEM(temp_tp, 0);
  jac_x = (PyArrayObject *) PyTuple_GET_ITEM(temp_tp, 1);
  Py_INCREF(fun_x);
  Py_INCREF(jac_x);
  Py_DECREF(temp_tp);
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
  // PyArrayObject * for the packed lower Cholesky factor of the [modified]
  // Hessian and the computed descent direction
  PyArrayObject *pchol_x, *d_x;
  // step size chosen by Armijo rule backtracking line search scaling d_x
  double eta;
  // optimize while not converged, i.e. avg. gradient norm is >= tolerance and
  // we have not reached the maximum iteration limit
  while (npy_frob_norm(jac_x) >= gtol && n_iter < maxiter) {
    /**
     * @todo routine to check if hess_x is positive definite enough and to
     * modify it as needed. the lower triangular Cholesky factor is returned in
     * packed format (use dpptrf for Cholesky computation, copying the lower
     * triangular portion of the [updated] Hessian, which is overwritten).
     * will probably need to modify the declarations at the beginning of the
     * function to include PyArrayObject * for pchol_x, d_x, and a double *
     * for x_data, the double * to the data of x.
     */
    /*
    pchol_x = cholesky_spd_hessian_update(hess_x, beta);
    if (pchol_x == NULL) {
      goto except_hess_x;
    }
    */
   /**
    * @todo using pchol_x, the packed lower triangular Cholesky factor, solve
    * pchol_x @ y = jac_x and then pchol_x.T @ p = y, p the descent direction.
    * since pchol_x is in packed format, use dtptrs to solve both systems.
    */
    /*
    d_x = compute_mnewton_descent_packed(pchol_x, jac_x);
    if (d_x == NULL) {
      Py_DECREF(pchol_x);
      goto except_hess_x;
    }
    */
    // TODO: use Armijo rule to compute step size with backtracking line search
    /*
    eta = armijo_backtrack(fun, jac, fun_args, d_x, alpha, gamma);
    */
    // TODO: update values in x (break out into separate routine?)

    // clean up unneeded pchol_x, d_x
    /*
    Py_DECREF(pchol_x);
    Py_DECREF(d_x);
    */
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
    // else again assign temp_tp[0], temp_tp[1] to fun_x, jac_x + Py_INCREF.
    // don't need temp_tp anymore, so Py_DECREF it
    fun_x = PyTuple_GET_ITEM(temp_tp, 0);
    jac_x = (PyArrayObject *) PyTuple_GET_ITEM(temp_tp, 1);
    Py_INCREF(fun_x);
    Py_INCREF(jac_x);
    Py_DECREF(temp_tp);
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
  // optimization is completed so populate the OptimizeResult. NULL on error
  PyObject *res = populate_OptimizeResult(
    x, 1, 0, "Optimization terminated successfully.", fun_x, jac_x, hess_x,
    NULL, n_fev, n_jev, n_hev, n_iter, NULL
  );
  if (res == NULL) {
    goto except_hess_x;
  }
  // PyObject_SetAttrString internally Py_INCREFs, so clean up and return res
  Py_DECREF(hess_x);
  Py_DECREF(jac_x);
  Py_DECREF(fun_x);
  Py_DECREF(fun_args);
  Py_DECREF(x);
  return res;
// clean up on error
except_hess_x:
  Py_DECREF(hess_x);
except_fun_jac_x:
  Py_DECREF(jac_x);
  Py_DECREF(fun_x);
except_fun_args:
  Py_DECREF(fun_args);
except_x:
  Py_DECREF(x);
  return NULL;
}

// _mnewton methods
static PyMethodDef _mnewton_methods[] = {
  {
    "mnewton", (PyCFunction) mnewton,
    METH_VARARGS | METH_KEYWORDS, mnewton_doc
  },
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
  "The mnewton function can be used as a frontend for scipy.optimize.minimize."
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
  // PyObject * for module and capsule, void * array for C API (static!)
  PyObject *module, *c_api_obj;
  static void *Py__mnewton_API[Py__mnewton_API_pointers];
  // import NumPy Array C API. automatically returns NULL on error.
  import_array();
  // create module, NULL on error
  module = PyModule_Create(&_mnewton_module);
  if (module == NULL) {
    return NULL;
  }
  // initialize the pointers for the C function pointer API
  Py__mnewton_API[Py__mnewton_remove_specified_kwargs_NUM] = \
    (void *) remove_specified_kwargs;
  Py__mnewton_API[Py__mnewton_remove_unspecified_kwargs_NUM] = \
    (void *) remove_unspecified_kwargs;
  Py__mnewton_API[Py__mnewton_npy_frob_norm_NUM] = \
    (void *) npy_frob_norm;
  Py__mnewton_API[Py__mnewton_tuple_prepend_single_NUM] = \
    (void *) tuple_prepend_single;
  Py__mnewton_API[Py__mnewton_compute_loss_grad_NUM] = \
    (void *) compute_loss_grad;
  Py__mnewton_API[Py__mnewton_compute_hessian_NUM] = \
    (void *) compute_hessian;
  Py__mnewton_API[Py__mnewton_populate_OptimizeResult_NUM] = \
    (void *) populate_OptimizeResult;
  // create capsule containing address to C array API. PyModule_AddObject only
  // steals ref on success, so we have to XDECREF, DECREF as needed on error
  c_api_obj = PyCapsule_New(
    (void *) Py__mnewton_API, "npy_lapacke_demo.solvers._mnewton._C_API", NULL
  );
  if (PyModule_AddObject(module, "_C_API", c_api_obj) < 0) {
    // need to Py_XDECREF since c_api_obj may be NULL
    Py_XDECREF(c_api_obj);
    Py_DECREF(module);
    return NULL;
  }
  return module;
}