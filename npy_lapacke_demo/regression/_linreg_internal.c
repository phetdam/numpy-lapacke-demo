/**
 * @file _linreg_internal.c
 * @author Derek Huang <djh458@stern.nyu.edu>
 * @brief Wrappers for internal C functions in `_linreg.c`.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// gives access to internal C functions in _linreg.c
#include "linreginternal.h"

// docstring for npy_vector_matrix_mean wrapper
PyDoc_STRVAR(
  npy_vector_matrix_mean_doc,
  "npy_vector_matrix_mean(ar)"
  "\n--\n\n"
  "Python-accessible wrapper for internal function npy_vector_matrix_mean."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "ar : numpy.ndarray\n"
  "    Input array shape (n_rows,) or (n_rows, n_cols), convertible to\n"
  "    NPY_DOUBLE type. For safety, error checking is done. Exception will\n"
  "    be raised if ar is empty or of incorrect shape."
  "\n\n"
  "Returns\n"
  "-------\n"
  "float or numpy.ndarray\n"
  "    If ar has shape (n_rows,), a Python float is returned (flat mean of\n"
  "    the elements, while if ar has shape (n_rows, n_cols), a numpy.ndarray\n"
  "    shape (n_cols,) is returned, giving the mean across the rows, i.e.\n"
  "    like calling ar.mean(axis=0) in Python."
);
/**
 * Python-accessible wrapper for `npy_vector_matrix_mean`.
 * 
 * @param self `PyObject *` module (unused)
 * @param arg `PyObject *` single argument. Method uses `METH_O` flag in its
 *     `PyMethodDef` in `_linreg_methods`, so no `PyArg_ParseTuple` needed.
 * @returns New reference, either `PyArrayObject *` flat vector if `arg` can be
 *     converted to 2D `PyArrayObject *` with type `NPY_DOUBLE` or
 *     `PyFloatObject *` if `arg` can be converted to 1D `PyArrayObject *`.
 */
static PyObject *
npy_vector_matrix_mean(PyObject *self, PyObject *arg)
{
  // only one argument is expected, PyArrayObject *, so we directly convert
  PyArrayObject *ar;
  ar = (PyArrayObject *) PyArray_FROM_OTF(arg, NPY_DOUBLE, NPY_ARRAY_CARRAY);
  if (ar == NULL) {
    return NULL;
  }
  // cannot pass empty ndarray to npy_vector_matrix_mean or it'll crash
  if (PyArray_SIZE(ar) == 0) {
    PyErr_SetString(PyExc_ValueError, "ar must not be empty");
    goto except;
  }
  // otherwise, we can pass this to Py__linreg_npy_vector_matrix_mean
  PyObject *res = Py__linreg_npy_vector_matrix_mean(ar);
  // if res is NULL, we can propagate this. have to Py_DECREF ar anyways
  Py_DECREF(ar);
  return res;
// clean up ar on exceptions
except:
  Py_DECREF(ar);
  return NULL;
}

// docstring for compute_intercept wrapper
PyDoc_STRVAR(
  compute_intercept_doc,
  "compute_intercept(coef, x_mean, y_mean)"
  "\n--\n\n"
  "Python-accessible wrapper for internal function compute_intercept."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "coef : numpy.ndarray\n"
  "    Linear model coefficients, shape (n_features,) for single-target\n"
  "    problems, (n_targets, n_features) for multi-target problems.\n"
  "    Should be convertable to NPY_DOUBLE, NPY_ARRAY_IN_ARRAY flags.\n"
  "x_mean : numpy.ndarray\n"
  "    Mean of the input rows, shape (n_features,). Should also be\n"
  "    convertible to the NPY_DOUBLE, NPY_ARRAY_IN_ARRAY flags.\n"
  "y_mean : float or numpy.ndarray\n"
  "    Mean of the response rows, either a float in the single-target case\n"
  "    or a numpy.ndarray in the multi-target case shape (n_targets,). If\n"
  "    array, should be convertible to NPY_DOUBLE, flags NPY_ARRAY_IN_ARRAY."
  "\n\n"
  "Returns\n"
  "-------\n"
  "float or numpy.ndarray\n"
  "    If coef has shape (n_features,) while y_mean is a float, a Python\n"
  "    float is returned, while if coef has shape (n_targets, n_features)\n"
  "    and y_mean has shape (n_targets,), then the function returns a\n"
  "    numpy.ndarray shape (n_targets,) instead."
);
/**
 * Python-accessible wrapper for `compute_intercept`.
 * 
 * @param self `PyObject *` module (unused)
 * @param arg `PyObject *` positional args
 * @returns New reference, either `PyArrayObject *` flat vector if the response
 *     is multi-target or a `PyFloatObject *` if response is single-target.
 *     `PyArrayObject *` has `NPY_DOUBLE` type and `NPY_ARRAY_CARRAY` flags.
 */
static PyObject *
compute_intercept(PyObject *self, PyObject *args)
{
  // PyArrayObject * for coefficients, mean of input matrix
  PyArrayObject *coef, *x_mean;
  coef = x_mean = NULL;
  // PyObject * for mean of the response, could be float or ndarray
  PyObject *y_mean = NULL;
  // parse using PyArg_ParseTuple; note only y_mean is not Py_INCREF'd so we
  // Py_XINCREF it since at except, we Py_XDECREF y_mean
  if (
    !PyArg_ParseTuple(
      args, "O&O&O", PyArray_Converter, (void *) &coef,
      PyArray_Converter, (void *) &x_mean, &y_mean
    )
  ) {
    Py_XINCREF(y_mean);
    goto except;
  }
  // Py_INCREF y_mean so we can Py_XDECREF coef, x_mean, y_mean all at once on
  // cleanup instead of having to handle the y_mean case separately
  Py_INCREF(y_mean);
  // temp to hold the conversion result of coef, x_mean, y_mean
  PyArrayObject *temp_ar;
  // convert coef to NPY_DOUBLE with NPY_ARRAY_IN_ARRAY flags
  temp_ar = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) coef, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  if (temp_ar == NULL) {
    goto except;
  }
  // on success, Py_DECREF coef and assign temp_ar to coef
  Py_DECREF(coef);
  coef = temp_ar;
  // convert x_mean to NPY_DOUBLE with NPY_ARRAY_IN_ARRAY flags
  temp_ar = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) x_mean, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  if (temp_ar == NULL) {
    goto except;
  }
  // on success, Py_DECREF x_mean and assign temp_ar to x_mean
  Py_DECREF(x_mean);
  x_mean = temp_ar;
  // neither coef and x_mean may be empty
  if (PyArray_SIZE(coef) == 0) {
    PyErr_SetString(PyExc_ValueError, "coef must not be empty");
    goto except;
  }
  if (PyArray_SIZE(x_mean) == 0) {
    PyErr_SetString(PyExc_ValueError, "x_mean must not be empty");
    goto except;
  }
  // check that coef is either 1 or 2 dims, x_mean is 1 dim
  if (PyArray_NDIM(coef) != 1 && PyArray_NDIM(coef) != 2) {
    PyErr_SetString(PyExc_ValueError, "coef must either be 1D or 2D");
    goto except;
  }
  if (PyArray_NDIM(x_mean) != 1) {
    PyErr_SetString(PyExc_ValueError, "x_mean must be 1D");
    goto except;
  }
  // get n_targets, n_features from coef
  npy_intp n_features, n_targets;
  // ndims == 1 => single-target, else multi-target
  if (PyArray_NDIM(coef) == 1) {
    n_features = PyArray_DIM(coef, 0);
    n_targets = 1;
  }
  else {
    n_features = PyArray_DIM(coef, 1);
    n_targets = PyArray_DIM(coef, 0);
  }
  // check that x_mean has length n_features
  if (PyArray_DIM(x_mean, 0) != n_features) {
    PyErr_SetString(PyExc_ValueError, "x_mean must have shape (n_features,)");
    goto except;
  }
  // if n_targets == 1, check that y_mean is a PyFloatObject *
  if (n_targets == 1) {
    if (!PyFloat_Check(y_mean)) {
      PyErr_SetString(
        PyExc_TypeError, "y_mean must be float in single-target case"
      );
      goto except;
    }
  }
  // else convert y_mean to PyArrayObject * and check its shape
  else {
    temp_ar = (PyArrayObject *) PyArray_FROM_OTF(
      y_mean, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    if (temp_ar == NULL) {
      goto except;
    }
    // on success, Py_DECREF y_mean and set to temp_ar
    Py_DECREF(y_mean);
    y_mean = (PyObject *) temp_ar;
    // check that y_mean has 1 dimension only
    if (PyArray_NDIM((PyArrayObject *) y_mean) != 1) {
      PyErr_SetString(PyExc_ValueError, "y_mean must be 1D");
      goto except;
    }
    // check that y_mean has shape (n_targets,)
    if (PyArray_DIM((PyArrayObject *) y_mean, 0) != n_targets) {
      PyErr_SetString(PyExc_ValueError, "y_mean must have shape (n_targets,)");
      goto except;
    }
  }
  // call Py__linreg_compute_intercept. if NULL, we propagate to return
  PyObject *res = Py__linreg_compute_intercept(coef, x_mean, y_mean);
  // Py_DECREF coef, x_mean, y_mean and return
  Py_DECREF(coef);
  Py_DECREF(x_mean);
  Py_DECREF(y_mean);
  return res;
// clean up  coef, x_mean, y_mean on exceptions
except:
  Py_XDECREF(coef);
  Py_XDECREF(x_mean);
  Py_XDECREF(y_mean);
  return NULL;
}

// _linreg_internal methods, possibly including EXTERNAL_* wrappers
static PyMethodDef _linreg_internal_methods[] = {
  {
    "npy_vector_matrix_mean",
    (PyCFunction) npy_vector_matrix_mean,
    METH_O,
    npy_vector_matrix_mean_doc
  },
  {
    "compute_intercept",
    (PyCFunction) compute_intercept,
    METH_VARARGS,
    compute_intercept_doc
  },
  // sentinel marking end of array
  {NULL, NULL, 0, NULL}
};

// _linreg_internal module definition
static PyModuleDef _linreg_internal_module = {
  PyModuleDef_HEAD_INIT,
  // name, docstring, size = -1 to disable subinterpreter support, methods
  .m_name = "_linreg_internal",
  .m_doc = "Wrappers for unit testing internal C functions in _linreg.",
  .m_size = -1,
  .m_methods = _linreg_internal_methods
};

// module initialization function
PyMODINIT_FUNC
PyInit__linreg_internal(void)
{
  // import NumPy Array C API. automatically returns NULL on error.
  import_array();
  // import _linreg C API. automatically returns NULL on error.
  import__linreg();
  // create module and return. NULL on error
  return PyModule_Create(&_linreg_internal_module);
}