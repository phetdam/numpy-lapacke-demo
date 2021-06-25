/**
 * @file _linreg.c
 * @brief C implementation of OLS linear regression solved by QR or SVD.
 * 
 * Directly calls into LAPACKE routines dgelsy, dgelss to solve.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

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

// struct representing our linear regression estimator
typedef struct {
  PyObject_HEAD
  // 1 if model will fit the intercept, 0 otherwise
  char fit_intercept;
  // name of the solution method
  const char *solver;
  // coefficients and intercept of the linear model
  PyObject *coef_, *intercept_;
  /**
   * effective rank and singular values of the data matrix. singular_ is NULL
   * unless we solve using SVD. trailing underscore follows sklearn convention.
   * use long to support very large matrices when linked with ILP64 Intel MKL
   * or a 64-bit version of OpenBLAS, i.e. built with INTERFACE64=1
   */
  long rank_;
  PyObject *singular_;
  // private attribute. 1 when model is fitted, 0 otherwise. the members with
  // trailing underscores are accessible iff fitted is 1.
  char fitted;
} LinearRegression;

/**
 * Computes mean across rows for 2D ndarray or scalar mean for 1D ndarray.
 * 
 * Do NOT call without proper argument checking. Could use `PyArray_Mean` but
 * since this function is specialized for 2D arrays it's faster to do manually.
 * 
 * @param ar `PyArrayObject *` ndarray to operate on. Must have type
 *     `NPY_DOUBLE`, flags `NPY_ARRAY_CARRAY` (row-major ordering), nonempty.
 * @returns New `PyObject *` reference, either 1D `PyArrayObject *` in case
 *     `ar` is 2D or `PyFLoatObject *` in case `ar` is 1D. `NULL` on error +
 *     exception set. If `PyArrayObject *`, has `NPY_ARRAY_CARRAY` flags.
 */
static PyObject *
npy_vector_matrix_mean(PyArrayObject *ar)
{
  // data pointer to ar
  double *data = (double *) PyArray_DATA(ar);
  // PyObject * for mean. either PyArrayObject * or PyFloatObject *
  PyObject *mean_o;
  // if 1D array, just compute flat mean and use PyFloatObject *
  if (PyArray_NDIM(ar) == 1) {
    double mean_d = 0;
    for (npy_intp i = 0; i < PyArray_SIZE(ar); i++) {
      mean_d += data[i];
    }
    mean_o = PyFloat_FromDouble(mean_d / (double) PyArray_SIZE(ar));
    // on error, mean_o is NULL; it will be returned after we break
  }
  // else if 2D array
  else if (PyArray_NDIM(ar) == 2) {
    // get number of rows, cols
    npy_intp n_rows, n_cols;
    n_rows = PyArray_DIM(ar, 0);
    n_cols = PyArray_DIM(ar, 1);
    // dims for the mean of the rows
    npy_intp mean_dims[] = {n_cols};
    // use ndarray to hold the output. has NPY_ARRAY_CARRAY flags
    mean_o = PyArray_SimpleNew(1, mean_dims, NPY_DOUBLE);
    if (mean_o == NULL) {
      return NULL;
    }
    // data pointer for mean_ar
    double *mean_data = (double *) PyArray_DATA((PyArrayObject *) mean_o);
    // compute means down the rows of ar. on completion, return
    for (npy_intp j = 0; j < n_cols; j++) {
      // compute column mean
      double mean_j = 0;
      for (npy_intp i = 0; i < n_rows; i++) {
        mean_j += data[i * n_cols + j];
      }
      mean_data[j] = mean_j / n_rows;
    }
  }
  // else error, not intended for ndarrays with more dimensions
  else {
    PyErr_SetString(PyExc_ValueError, "ar must be 1D or 2D only");
    return NULL;
  }
  // return mean_o; could be NULL if error
  return mean_o;
}

/**
 * wrapper code for npy_vector_matrix_mean that lets us test it from Python.
 * note that __INTELLISENSE__ is always defined in VS Code, so including the
 * defined(__INTELLISENSE__) lets Intellisense work on the code in VS Code.
 */
#if defined(__INTELLISENSE__) || defined(EXPOSE_INTERNAL)
// docstring for EXPOSED_npy_vector_matrix_mean
PyDoc_STRVAR(
  EXPOSED_npy_vector_matrix_mean_doc,
  "EXPOSED_npy_vector_matrix_mean(ar)"
  "\n--\n\n"
  EXPOSE_INTERNAL_NOTICE
  "\n\n"
  "Python-accessible wrapper for internal function ``npy_vector_matrix_mean``."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "ar : numpy.ndarray\n"
  "    Input array shape ``(n_rows,)`` or ``(n_rows, n_cols)``, convertable\n"
  "    to ``NPY_DOUBLE`` type. For safety, error checking is done. Exception\n"
  "    will be raised if ``ar`` is empty or of incorrect shape."
  "\n\n"
  "Returns\n"
  "-------\n"
  "float or numpy.ndarray\n"
  "    If ``ar`` has shape ``(n_rows,)``, a Python float is returned (flat\n"
  "    mean of the elements, while if ``ar`` has shape ``(n_rows, n_cols)``,\n"
  "    a :class:`numpy.ndarray` shape ``(n_cols,)`` is returned, giving the\n"
  "    mean across the rows, i.e. like ``ar.mean(axis=0)``."
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
EXPOSED_npy_vector_matrix_mean(PyObject *self, PyObject *arg)
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
  // otherwise, we can pass this to npy_vector_matrix_mean
  PyObject *res = npy_vector_matrix_mean(ar);
  // if res is NULL, we can propagate this. have to Py_DECREF ar anyways
  Py_DECREF(ar);
  return res;
// clean up ar on exceptions
except:
  Py_DECREF(ar);
  return NULL;
}
#endif /* defined(__INTELLISENSE__) || defined(EXPOSE_INTERNAL) */

/**
 * Computes the intercept for a linear regression model.
 * 
 * Do NOT call without proper input checking. Inner product computed manually
 * in the single-target case or using `cblas_dgemv` in multi-target case.
 * 
 * @param coef `PyArrayObject *` linear coefficients computed by a solver,
 *     shape `(n_targets, n_features)` or `(n_features,)`. As usual, must have
 *     type `NPY_DOUBLE` and the `NPY_ARRAY_IN_ARRAY` flags. Note that it is
 *     NOT guaranteed to be writeable, although aligned and C-contiguous.
 * @param x_mean `PyArrayObject *` sample mean of the input matrix rows. This
 *     must have shape `(n_features,)`, same type and flags as `coef`.
 * @param y_mean `PyObject *`, either `PyArrayObject *` sample mean of the
 *     response rows, shape `(n_targets,)`, else `PyFloatObject *`. If
 *     `PyFloatObject *`, then `coef` must have shape `(n_features,)`.
 * @returns `PyObject *`, either a `PyFloatObject *` if single-target, else
 *     `PyArrayObject *` shape `(n_targets,)` if multi-target.
 */
static PyObject *
compute_intercept(PyArrayObject *coef, PyArrayObject *x_mean, PyObject *y_mean)
{
  // get n_targets, n_features from coef. handle single and multi-target cases
  npy_intp n_features, n_targets;
  if (PyArray_NDIM(coef) == 1) {
    n_features = PyArray_DIM(coef, 0);
    n_targets = 1;
  }
  else {
    n_features = PyArray_DIM(coef, 1);
    n_targets = PyArray_DIM(coef, 0);
  }
  // get data pointers for coef, x_mean
  double *coef_data = (double *) PyArray_DATA(coef);
  double *x_mean_data = (double *) PyArray_DATA(x_mean);
  // PyObject * pointing the intercept we will return
  PyObject *bias;
  // compute intercept. for single-target case, compute manually.
  if (n_targets == 1) {
    // inner product of coef, x_mean + y_mean as double (y_mean is python float)
    double wx_d, y_mean_d;
    // convert y_mean to double. on error, no Py_DECREF since refs borrowed
    y_mean_d = PyFloat_AsDouble(y_mean);
    if (PyErr_Occurred()) {
      return NULL;
    }
    // compute wx_d, the inner product of coef and x_mean
    wx_d = 0;
    for (npy_intp i = 0; i < n_features; i++) {
      wx_d += coef_data[i] * x_mean_data[i];
    }
    // assign PyFloatObject * from y_mean_d - wx_d to bias. NULL on error.
    bias = PyFloat_FromDouble(y_mean_d - wx_d);
  }
  // for multi-target case, use dgemv to compute matrix-vector product
  else {
    // allocate new intercept array shape (n_targets,) for bias using y_mean.
    // we need to copy y_mean to use dgemv, hence NPY_ARRAY_ENSURECOPY.
    bias = PyArray_FROM_OTF(
      y_mean, NPY_DOUBLE, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY
    );
    // return NULL on error since all refs borrowed, nothing to Py_DECREF
    if (bias == NULL) {
      return NULL;
    }
    // pointer to data of the intercept, shape (n_targets,)
    double *bias_data = (double *) PyArray_DATA((PyArrayObject *) bias);
    // compute intercept, stored in bias_data, using coef_data, x_mean_data,
    // bias_data, i.e. -coef_data @ x_mean_data + bias_data
    cblas_dgemv(
      CblasRowMajor, CblasNoTrans,
      (const MKL_INT) n_targets, (const MKL_INT) n_features, -1,
      (const double *) coef_data, (const MKL_INT) n_features,
      (const double *) x_mean_data, 1, 1, bias_data, 1
    );
    // done, leave the if-else block
  }
  // return bias. NULL on error, which we propagate
  return bias;
}

/**
 * wrapper code for compute_intercept that lets us test it from Python.
 * note that __INTELLISENSE__ is always defined in VS Code, so including the
 * defined(__INTELLISENSE__) lets Intellisense work on the code in VS Code.
 */
#if defined(__INTELLISENSE__) || defined(EXPOSE_INTERNAL)
// docstring for EXPOSED_compute_intercept
PyDoc_STRVAR(
  EXPOSED_compute_intercept_doc,
  "EXPOSED_compute_intercept(coef, x_mean, y_mean)"
  "\n--\n\n"
  EXPOSE_INTERNAL_NOTICE
  "\n\n"
  "Python-accessible wrapper for internal function ``compute_intercept``."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "coef : numpy.ndarray\n"
  "    Linear model coefficients, shape ``(n_features,)`` for single-target\n"
  "    problems, ``(n_targets, n_features)`` for multi-target problems.\n"
  "    Should be convertable to ``NPY_DOUBLE``, ``NPY_ARRAY_IN_ARRAY`` flags.\n"
  "x_mean : numpy.ndarray\n"
  "    Mean of the input rows, shape ``(n_features,)``. Should also be\n"
  "    convertible to the ``NPY_DOUBLE``, ``NPY_ARRAY_IN_ARRAY`` flags.\n"
  "y_mean : float or numpy.ndarray\n"
  "    Mean of the response rows, either a float in the single-target case\n"
  "    or a :class:`numpy.ndarray` in the multi-target case shape\n"
  "    ``(n_targets,)``. If array, should also be convertible to\n"
  "    ``NPY_DOUBLE`` with ``NPY_ARRAY_IN_ARRAY`` flags."
  "\n\n"
  "Returns\n"
  "-------\n"
  "float or numpy.ndarray\n"
  "    If ``coef`` has shape ``(n_features,)`` while ``y_mean`` is a float,\n"
  "    a Python float is returned, while if ``coef`` has shape\n"
  "    ``(n_targets, n_features)`` and ``y_mean`` has shape ``(n_targets,)``,\n"
  "    a :class:`numpy.ndarray` shape ``(n_targets,)`` is returned."
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
EXPOSED_compute_intercept(PyObject *self, PyObject *args)
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
  // get result from compute_intercept. if NULL, we propagate to return
  PyObject *res = compute_intercept(coef, x_mean, y_mean);
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
#endif /* defined(__INTELLISENSE__) || defined(EXPOSE_INTERNAL) */

/**
 * QR solver for the `LinearRegression` class.
 * 
 * Do NOT call without proper argument checking. Copies input matrix.
 * 
 * `coef_` attribute will point to `PyArrayObject *`, `intercept_` attribute
 * will point to `PyArrayObject *` if multi-target else a `PyFloatObject *`
 * for single-target, and `singular_` will be `Py_None`. `fitted` set to `1`.
 * 
 * Note we don't need an `EXPOSED_*` method for `qr_solver` since we can
 * verify its performance by fitting a `LinearRegression` instance with
 * `solver="qr"`. We should not expect `qr_solver` to result in  errors.
 * 
 * @param self `LinearRegression *` instance
 * @param input_ar `PyArrayObject *` input, shape `(n_samples, n_features)`
 * @param output_ar `PyArrayObject *` response, shape `(n_samples,)` for single
 *     output or shape `(n_samples, n_targets)` for multi-output
 * @returns `0` on success, `-1` on error.
 */
static int
qr_solver(
  LinearRegression *self,
  PyArrayObject *input_ar, PyArrayObject *output_ar
)
{
  // get number of samples, features, and targets
  npy_intp n_samples, n_features, n_targets;
  n_samples = PyArray_DIM(input_ar, 0);
  n_features = PyArray_DIM(input_ar, 1);
  n_targets = (PyArray_NDIM(output_ar) == 1) ? 1 : PyArray_DIM(output_ar, 1);
  // centered copy of input_ar actually used in calculation
  PyArrayObject *input_cent_ar = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) input_ar, NPY_DOUBLE, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY
  );
  if (input_cent_ar == NULL) {
    return -1;
  }
  // copy of output_ar, used to store the result
  PyArrayObject *output_copy_ar = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) output_ar, NPY_DOUBLE, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY
  );
  if (output_copy_ar == NULL) {
    goto except_input_cent_ar;
  }
  /**
   * we need to center input_cent_ar and need the means of the rows of input_ar,
   * output_ar to compute the intercept, so we compute them now. the mean of
   * the of the input_ar rows is guaranteed to be PyArrayObject *, but the
   * mean of the rows of output_ar might be PyFloatObject * if n_targets = 1.
   */
  PyArrayObject *input_mean;
  input_mean = (PyArrayObject *) npy_vector_matrix_mean(input_ar);
  if (input_mean == NULL) {
    goto except_output_copy_ar;
  }
  PyObject *output_mean;
  output_mean = npy_vector_matrix_mean(output_ar);
  if (output_mean == NULL) {
    goto except_input_mean;
  }
  // double * to the data of input_cent_ar, output_copy_ar, input_mean
  double *input_cent_data = (double *) PyArray_DATA(input_cent_ar);
  double *output_copy_data = (double *) PyArray_DATA(output_copy_ar);
  double *input_mean_data = (double *) PyArray_DATA(input_mean);
  // center input_cent_data by subtracting out the means for each column
  for (npy_intp j = 0; j < n_features; j++) {
    for (npy_intp i = 0; i < n_samples; i++) {
      input_cent_data[i * n_features + j] -= input_mean_data[j];
    }
  }
  // temp to hold information on the pivoted columns of input_cent_data.
  // we use PyMem_RawMalloc so the interpreter can track all the memory used.
  lapack_int *pivot_idx;
  pivot_idx = (lapack_int *) PyMem_RawMalloc(n_features * sizeof(lapack_int));
  if (pivot_idx == NULL) {
    goto except_output_mean;
  }
/**
 * typically npy_intp has same size as long, 64-bit on 64-bit architecture.
 * if we aren't linked against Intel MKL (make links to ILP64, 64-bit int),
 * i.e. MKL_ILP64 not defined, and not linked against OpenBLAS built with
 * INTERFACE64=1 for 64-bit int, i.e. OPENBLAS_USE64BITINT not defined, we need
 * to check if n_samples, n_features, n_targets exceeds INT_MAX. if so, raise
 * raise OverflowError, as values > 2^32 - 1 can cause dangerous results.
 */ 
#if !defined(MKL_ILP64) && !defined(OPENBLAS_USE64BITINT)
  if (n_samples > INT_MAX || n_features > INT_MAX || n_targets > INT_MAX) {
    PyErr_SetString(
      PyExc_OverflowError,
      "CBLAS/LAPACKE implementation does not support 64-bit indexing"
    );
    goto except_pivot_idx;
  }
#endif
  // status returned by dgelsy. if -i returned, the ith input is illegal
  lapack_int qr_status;
  /**
   * call dgelsy on input_cent_data, output_copy_data, pivot_idx which solves
   * linear system using QR decomposition. all arrays overwritten, and the
   * result is stored in output_copy_ar. LAPACKE_dgelsy returns < 0 => error.
   * arbitrarily choose 1e-8 as reciprocal of max condition of input_cent_data.
   */
  lapack_int rank_;
  qr_status = LAPACKE_dgelsy(
    LAPACK_ROW_MAJOR, (lapack_int) n_samples, (lapack_int) n_features,
    (lapack_int) n_targets, input_cent_data, (lapack_int) n_features,
    output_copy_data, (lapack_int) n_targets, pivot_idx, 1e-8, &rank_
  );
  // if qr_status < 0, error, so we need to set an exception
  if (qr_status < 0) {
    PyErr_Format(
      PyExc_RuntimeError, "LAPACKE_dgelsy: parameter %ld is an illegal value. "
      "please ensure X, y do not contain nan or inf values", -qr_status
    );
    goto except_pivot_idx;
  }
  // dims for coef_. if multi-target, shape (n_targets, n_features), while if
  // single-target, shape (n_targets,), so different initializations.
  npy_intp coef_dims[2];
  if (n_targets == 1) {
    coef_dims[0] = n_features;
  }
  else {
    coef_dims[0] = n_targets;
    coef_dims[1] = n_features;
  }
  // create new NPY_ARRAY_CARRAY to hold the solution in output_copy_ar. if
  // single target, ignore second (unset) dim, else use both for multi
  PyArrayObject *coef_ar = (PyArrayObject *) PyArray_SimpleNew(
    (n_targets == 1) ? 1 : 2, coef_dims, NPY_DOUBLE
  );
  if (coef_ar == NULL) {
    goto except_pivot_idx;
  }
  // copy data from output_copy_data into coef_ar's data pointer (transpose)
  double *coef_data = (double *) PyArray_DATA(coef_ar);
  for (npy_intp i = 0; i < n_targets; i++) {
    for (npy_intp j = 0; j < n_features; j++) {
      coef_data[i * n_features + j] = output_copy_data[j * n_targets + i];
    }
  }
  // set self->rank_ and self->coef_
  self->rank_ = (long) rank_;
  self->coef_ = (PyObject *) coef_ar;
  // if self->fit_intercept is 0, set fit_intercept to new PyFloatObject *
  if (!self->fit_intercept) {
    self->intercept_ = PyFloat_FromDouble(0.);
    if (self->intercept_ == NULL) {
      goto except_coef_ar;
    }
  }
  // else we compute the intercept using coef_ar, input_mean, output_mean
  else {
    self->intercept_ = compute_intercept(
      (PyArrayObject *) self->coef_, input_mean, output_mean
    );
    if (self->intercept_ == NULL) {
      goto except_coef_ar;
    }
  }
  // singular_ always None when using QR solver
  Py_INCREF(Py_None);
  self->singular_ = Py_None;
  // coef_, intercept_, rank_, singular_ have been set. first free pivot_idx
  PyMem_RawFree((void *) pivot_idx);
  // clean up PyObject * previously allocated (except coef_ar)
  Py_DECREF(output_mean);
  Py_DECREF(input_mean);
  Py_DECREF(output_copy_ar);
  Py_DECREF(input_cent_ar);
  // done with cleanup, so set self->fitted to 1 and we can return
  self->fitted = 1;
  return 0;
// clean up all the allocated PyObject * and memory on failure
except_coef_ar:
  Py_DECREF(coef_ar);
except_pivot_idx:
  PyMem_RawFree((void *) pivot_idx);
except_output_mean:
  Py_DECREF(output_mean);
except_input_mean:
  Py_DECREF(input_mean);
except_output_copy_ar:
  Py_DECREF(output_copy_ar);
except_input_cent_ar:
  Py_DECREF(input_cent_ar);
  return -1;
}

/**
 * SVD solver for the `LinearRegression` class.
 * 
 * Do NOT call without proper argument checking. Copies input matrix.
 * 
 * `coef_` attribute will point to `PyArrayObject *`, `intercept_` attribute
 * will point to `PyArrayObject *` if multi-target else a `PyFloatObject *`
 * for single-target, and `singular_` will be `Py_None`. `fitted` set to `1`.
 * 
 * Note we don't need an `EXPOSED_*` method for `svd_solver` since we can
 * verify its performance by fitting a `LinearRegression` instance with
 * `solver="svd"`. We should not expect `svd_solver` to result in  errors.
 * 
 * Most of the code before and after the call to `dgelss` is essentially copied
 * from `qr_solver` since the setup is relatively similar.
 * 
 * @param self `LinearRegression *` instance
 * @param input_ar `PyArrayObject *` input, shape `(n_samples, n_features)`
 * @param output_ar `PyArrayObject *` response, shape `(n_samples,)` for single
 *     output or shape `(n_samples, n_targets)` for multi-output
 * @returns `0` on success, `-1` on error.
 */
static int
svd_solver(
  LinearRegression *self,
  PyArrayObject *input_ar, PyArrayObject *output_ar
)
{
  // get number of samples, features, and targets
  npy_intp n_samples, n_features, n_targets;
  n_samples = PyArray_DIM(input_ar, 0);
  n_features = PyArray_DIM(input_ar, 1);
  n_targets = (PyArray_NDIM(output_ar) == 1) ? 1 : PyArray_DIM(output_ar, 1);
  // centered copy of input_ar actually used in calculation
  PyArrayObject *input_cent_ar = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) input_ar, NPY_DOUBLE, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY
  );
  if (input_cent_ar == NULL) {
    return -1;
  }
  // copy of output_ar, used to store the result
  PyArrayObject *output_copy_ar = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) output_ar, NPY_DOUBLE, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY
  );
  if (output_copy_ar == NULL) {
    goto except_input_cent_ar;
  }
  /**
   * we need to center input_cent_ar and need the means of the rows of input_ar,
   * output_ar to compute the intercept, so we compute them now. the mean of
   * the of the input_ar rows is guaranteed to be PyArrayObject *, but the
   * mean of the rows of output_ar might be PyFloatObject * if n_targets = 1.
   */
  PyArrayObject *input_mean;
  input_mean = (PyArrayObject *) npy_vector_matrix_mean(input_ar);
  if (input_mean == NULL) {
    goto except_output_copy_ar;
  }
  PyObject *output_mean;
  output_mean = npy_vector_matrix_mean(output_ar);
  if (output_mean == NULL) {
    goto except_input_mean;
  }
  // double * to the data of input_cent_ar, output_copy_ar, input_mean
  double *input_cent_data = (double *) PyArray_DATA(input_cent_ar);
  double *output_copy_data = (double *) PyArray_DATA(output_copy_ar);
  double *input_mean_data = (double *) PyArray_DATA(input_mean);
  // center input_cent_data by subtracting out the means for each column
  for (npy_intp j = 0; j < n_features; j++) {
    for (npy_intp i = 0; i < n_samples; i++) {
      input_cent_data[i * n_features + j] -= input_mean_data[j];
    }
  }
  // allocate PyArrayObject * which will hold the singular values
  npy_intp singular_dims[] = {n_features};
  PyArrayObject *singular_ar = (PyArrayObject *) PyArray_SimpleNew(
    1, singular_dims, NPY_DOUBLE
  );
  if (singular_ar == NULL) {
    goto except_output_mean;
  }
  // pointer to data of singular_ar
  double *singular_data = (double *) PyArray_DATA(singular_ar);
/**
 * typically npy_intp has same size as long, 64-bit on 64-bit architecture.
 * if we aren't linked against Intel MKL (make links to ILP64, 64-bit int),
 * i.e. MKL_ILP64 not defined, and not linked against OpenBLAS built with
 * INTERFACE64=1 for 64-bit int, i.e. OPENBLAS_USE64BITINT not defined, we need
 * to check if n_samples, n_features, n_targets exceeds INT_MAX. if so, raise
 * raise OverflowError, as values > 2^32 - 1 can cause dangerous results.
 */ 
#if !defined(MKL_ILP64) && !defined(OPENBLAS_USE64BITINT)
  if (n_samples > INT_MAX || n_features > INT_MAX || n_targets > INT_MAX) {
    PyErr_SetString(
      PyExc_OverflowError,
      "CBLAS/LAPACKE implementation does not support 64-bit indexing"
    );
    goto except_singular_ar;
  }
#endif
  // status returned by dgelss. if -i returned, the ith input is illegal,
  // while i > 0 returned indicates that convergence failed.
  lapack_int svd_status;
  /**
   * call dgelss on input_cent_data, output_copy_data, pivot_idx which solves
   * linear system using SVD. all arrays overwritten, and the result is stored
   * in output_copy_ar. LAPACKE_dgelss returns < 0 => bad input, > 0 => that
   * convergence failed. like with dgelsy, we arbitrarily choose 1e-8 as
   * reciprocal of max condition of input_cent_data.
   */
  lapack_int rank_;
  svd_status = LAPACKE_dgelss(
    LAPACK_ROW_MAJOR, (lapack_int) n_samples, (lapack_int) n_features,
    (lapack_int) n_targets, input_cent_data, (lapack_int) n_features,
    output_copy_data, (lapack_int) n_targets, singular_data, 1e-8, &rank_
  );
  // handle the possible return values of dgelss. 0 is normal exit
  if (svd_status == 0) {
    ;
  }
  // svd_status < 0 => illegal input
  else if (svd_status < 0) {
    PyErr_Format(
      PyExc_RuntimeError, "LAPACKE_dgelss: parameter %ld is an illegal value. "
      "please ensure X, y do not contain nan or inf values", -svd_status
    );
    goto except_singular_ar;
  }
  // else svd_status > 0 => convergence failed
  else {
    PyErr_Format(
      PyExc_RuntimeError, "LAPACKE_dgelss: convergence failed. "
      "%ld off-diagonal elements of an intermediate bidiagonal form "
      "did not converge to zero", svd_status
    );
    goto except_singular_ar;
  }
  // dims for coef_. if multi-target, shape (n_targets, n_features), while if
  // single-target, shape (n_targets,), so different initializations.
  npy_intp coef_dims[2];
  if (n_targets == 1) {
    coef_dims[0] = n_features;
  }
  else {
    coef_dims[0] = n_targets;
    coef_dims[1] = n_features;
  }
  // create new NPY_ARRAY_CARRAY to hold the solution in output_copy_ar. if
  // single target, ignore second (unset) dim, else use both for multi
  PyArrayObject *coef_ar = (PyArrayObject *) PyArray_SimpleNew(
    (n_targets == 1) ? 1 : 2, coef_dims, NPY_DOUBLE
  );
  if (coef_ar == NULL) {
    goto except_singular_ar;
  }
  // copy data from output_copy_data into coef_ar's data pointer (tranpose)
  double *coef_data = (double *) PyArray_DATA(coef_ar);
  for (npy_intp i = 0; i < n_targets; i++) {
    for (npy_intp j = 0; j < n_features; j++) {
      coef_data[i * n_features + j] = output_copy_data[j * n_targets + i];
    }
  }
  // set self->rank_ and self->coef_
  self->rank_ = (long) rank_;
  self->coef_ = (PyObject *) coef_ar;
  // if self->fit_intercept is 0, set fit_intercept to new PyFloatObject *
  if (!self->fit_intercept) {
    self->intercept_ = PyFloat_FromDouble(0.);
    if (self->intercept_ == NULL) {
      goto except_coef_ar;
    }
  }
  // else we compute the intercept using coef_ar, input_mean, output_mean
  else {
    self->intercept_ = compute_intercept(
      (PyArrayObject *) self->coef_, input_mean, output_mean
    );
    if (self->intercept_ == NULL) {
      goto except_coef_ar;
    }
  }
  // set self->singular_, already set rank_, coef_, intercept_
  self->singular_ = (PyObject *) singular_ar;
  // coef_, intercept_, rank_, singular_ have been set. clean up all previously
  // allocated PyObject * except coef_ar, singular_ar
  Py_DECREF(output_mean);
  Py_DECREF(input_mean);
  Py_DECREF(output_copy_ar);
  Py_DECREF(input_cent_ar);
  // done with cleanup, so set self->fitted to 1 and we can return
  self->fitted = 1;
  return 0;
// clean up all the allocated PyObject * and memory on failure
except_coef_ar:
  Py_DECREF(coef_ar);
except_singular_ar:
  Py_DECREF(singular_ar);
except_output_mean:
  Py_DECREF(output_mean);
except_input_mean:
  Py_DECREF(input_mean);
except_output_copy_ar:
  Py_DECREF(output_copy_ar);
except_input_cent_ar:
  Py_DECREF(input_cent_ar);
  return -1;
}

/**
 * `__del__` method for the `LinearRegression` class.
 * 
 * @param self `PyObject *` to `LinearRegression` class instance.
 */
static void
LinearRegression_dealloc(LinearRegression *self)
{
  // need to decrement reference counts of owned objects. all might be NULL
  Py_XDECREF(self->coef_);
  Py_XDECREF(self->intercept_);
  Py_XDECREF(self->singular_);
  // finally, free the memory owned by the instance
  Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
 * `__new__` method for the `LinearRegression` class.
 * 
 * No checking of `type` since it is guaranteed to be `&LinearRegression_type`.
 * `args`, `kwargs` are ignored since `__init__` does initialization.
 * 
 * @param type `PyTypeObject *` to the `LinearRegression` type.
 * @param args `PyObject *` with positional args tuple (unused)
 * @param kwargs `PyObject *` with keyword args dict (unused)
 * @returns New `PyObject *` to a `LinearRegression` instance.
 */
static PyObject *
LinearRegression_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  // allocate new LinearRegression struct. NULL on error
  LinearRegression *self = (LinearRegression *) type->tp_alloc(type, 0); 
  if (self == NULL) {
    return NULL;
  }
  // initialize all the PyObject * slots to NULL. in Python, since we declare
  // the members with T_OBJECT_EX, access when NULL yields AttributeError
  self->coef_ = self->intercept_ = self->singular_ = NULL;
  // set default initializations for fit_intercept, solver
  self->fit_intercept = 1;
  self->solver = "svd";
  // fitted must start at zero and we can set rank_ to whatever we want
  self->fitted = self->rank_ = 0;
  return (PyObject *) self;
}

// keyword args that the LinearRegression __init__ method accepts
static char *LinearRegression_kwargs[] = {"fit_intercept", "solver"};
/**
 *  `__init__` method for the `LinearRegression` class.
 * 
 * @param self `LinearRegression *` to instance returned by `__new__`
 * @param args `PyObject *` positional args tuple
 * @param kwargs `PyObject *` keyword args dict
 * @returns `0` on success, `-1` on error with exception set
 */
static int
LinearRegression_init(LinearRegression *self, PyObject *args, PyObject *kwargs)
{
  // parse arguments. note that arguments are keyword only.
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "|$ps", LinearRegression_kwargs,
      &self->fit_intercept, &self->solver
    )
  ) {
    return -1;
  }
  // solver can only take "qr" or "svd". if neither, error + return -1
  if ((strcmp(self->solver, "qr") != 0) && (strcmp(self->solver, "svd") != 0)) {
    PyErr_SetString(
      PyExc_ValueError, "solver must be one of (\"qr\", \"svd\")"
    );
    return -1;
  }
  return 0;
}

// members of the LinearRegression type. note T_STRING implies READONLY.
static PyMemberDef LinearRegression_members[] = {
  {
    "fit_intercept", T_BOOL,
    offsetof(LinearRegression, fit_intercept), READONLY, NULL
  },
  {
    "solver", T_STRING,
    offsetof(LinearRegression, solver), READONLY, NULL
  },
  // sentinel marking end of array
  {NULL, 0, 0, 0, NULL}
};

/**
 * Getter for `coef_` attribute.
 * 
 * @param self `LinearRegression *` instance
 * @param closure `void *` (unused)
 * @returns `PyObject *` new reference to `coef_`.
 */
static PyObject *
LinearRegression_coef_getter(LinearRegression *self, void *closure)
{
  // if not fitted, raise AttributeError
  if (!self->fitted) {
    PyErr_SetString(PyExc_AttributeError, "coef_ only available after fitting");
    return NULL;
  }
  // else return coef_. note that fitted == 1 => self->coef_ != NULL.
  Py_INCREF(self->coef_);
  return self->coef_;
}

/**
 * Getter for `intercept_` attribute.
 * 
 * @param self `LinearRegression *` instance
 * @param closure `void *` (unused)
 * @returns `PyObject *` new reference to `intercept_`.
 */
static PyObject *
LinearRegression_intercept_getter(LinearRegression *self, void *closure)
{
  if (!self->fitted) {
    PyErr_SetString(
      PyExc_AttributeError, "intercept_ only available after fitting"
    );
    return NULL;
  }
  Py_INCREF(self->intercept_);
  return self->intercept_;
}

/**
 * Getter for `rank_` attribute.
 * 
 * @param self `LinearRegression *` instance
 * @param closure `void *` (unused)
 * @returns `PyLongObject *` cast to `PyObject *` constructed from `rank_`.
 */
static PyObject *
LinearRegression_rank_getter(LinearRegression *self, void *closure)
{
  if (!self->fitted) {
    PyErr_SetString(PyExc_AttributeError, "rank_ only available after fitting");
    return NULL;
  }
  // return new PyLongObject * from rank_. NULL on error
  return (PyObject *) PyLong_FromLong(self->rank_);
}

/**
 * Getter for `singular_` attribute.
 * 
 * @param self `LinearRegression *` instance
 * @param closure `void *` (unused)
 * @returns `PyObject *` new reference to `singular_`.
 */
static PyObject *
LinearRegression_singular_getter(LinearRegression *self, void *closure)
{
  if (!self->fitted) {
    PyErr_SetString(
      PyExc_AttributeError, "singular_ only available after fitting"
    );
    return NULL;
  }
  Py_INCREF(self->singular_);
  return self->singular_;
}

// members of the LinearRegression type accessed using getters/setters
static PyGetSetDef LinearRegression_getsets[] = {
  {"coef_", (getter) LinearRegression_coef_getter, NULL, NULL, NULL},
  {"intercept_", (getter) LinearRegression_intercept_getter, NULL, NULL, NULL},
  {"rank_", (getter) LinearRegression_rank_getter, NULL, NULL, NULL},
  {"singular_", (getter) LinearRegression_singular_getter, NULL, NULL, NULL},
  // sentinel marking end of array
  {NULL, NULL, NULL, NULL, NULL}
};

// docstring for the LinearRegression fit method
PyDoc_STRVAR(
  LinearRegression_fit_doc,
  "fit(X, y)"
  "\n--\n\n"
  "Fit an ordinary least squares linear regression model given ``X``, ``y``."
  "\n\n"
  "Returns ``self`` to allow method chaining. Note that ``X``, ``y`` will be\n"
  "copied if they are not of type :class:`numpy.ndarray`, not C-contiguous,\n"
  "not memory-aligned, and don't have ``dtype`` double."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "X : numpy.ndarray\n"
  "    Input array, shape ``(n_samples, n_features)``\n"
  "y : numpy.ndarray\n"
  "    Response array, shape ``(n_samples,)`` or ``(n_samples, n_targets)``"
  "\n\n"
  "Returns\n"
  "-------\n"
  "self"
);
/**
 * `fit` method for the `LinearRegression` class.
 * 
 * No keyword arguments needed, so `kwargs` is omitted.
 * 
 * @param self `LinearRegression *` instance
 * @param args `PyObject *` positional args tuple
 * @returns New `PyObject *` reference to `self` to allow method chaining.
 */
static PyObject *
LinearRegression_fit(LinearRegression *self, PyObject *args)
{
  // input ndarray and response ndarray. set to NULL so when we Py_XDECREF in
  // PyArg_ParseTuple we don't cause segfaults.
  PyArrayObject *input_ar, *output_ar;
  input_ar = output_ar = NULL;
  // parse input and response, converting to ndarray. must Py_XDECREF on error.
  if (
    !PyArg_ParseTuple(
      args, "O&O&", PyArray_Converter, (void *) &input_ar,
      PyArray_Converter, (void *) &output_ar
    )
  ) {
    goto except;
  }
  // check that input_ar and output_ar have positive size
  if (PyArray_SIZE(input_ar) < 1) {
    PyErr_SetString(PyExc_ValueError, "X must be nonempty");
    goto except;
  }
  if (PyArray_SIZE(output_ar) < 1) {
    PyErr_SetString(PyExc_ValueError, "y must be nonempty");
    goto except;
  }
  // check that input_ar and output_ar have appropriate shape
  if (PyArray_NDIM(input_ar) != 2) {
    PyErr_SetString(
      PyExc_ValueError, "X must have shape (n_samples, n_features)"
    );
    goto except;
  }
  if (PyArray_NDIM(output_ar) != 1 && PyArray_NDIM(output_ar) != 2) {
    PyErr_SetString(
      PyExc_ValueError,
      "y must have shape (n_samples,) or (n_samples, n_targets)"
    );
    goto except;
  }
  // get number of samples, number of features
  npy_intp n_samples, n_features;
  n_samples = PyArray_DIM(input_ar, 0);
  n_features = PyArray_DIM(input_ar, 1);
  // check that y has correct number of samples
  if (PyArray_DIM(output_ar, 0) != n_samples) {
    PyErr_SetString(PyExc_ValueError, "number of rows of X, y must match");
    goto except;
  }
  // check that n_samples >= n_features; required
  if (n_samples < n_features) {
    PyErr_SetString(PyExc_ValueError, "n_samples >= n_features required");
    goto except;
  }
  /**
   * convert to C-contiguous NPY_DOUBLE arrays. NPY_ARRAY_IN_ARRAY is same as
   * NPY_ARRAY_CARRAY but doesn't guarantee NPY_ARRAY_WRITEABLE. use temporary
   * PyObject * to hold the results of the cast each time. X, y will be copied
   * if they don't already satisfy the requirements.
   */
  PyArrayObject *temp_ar;
  // attempt conversion of input_ar
  temp_ar = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) input_ar, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  if (temp_ar == NULL) {
    goto except;
  }
  // on success, we can Py_DECREF input_ar and set input_ar to temp_ar
  Py_DECREF(input_ar);
  input_ar = temp_ar;
  // attempt conversion of output_ar
  temp_ar = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) output_ar, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  if (temp_ar == NULL) {
    goto except;
  }
  // on success, we can Py_DECREF output_ar and set output_ar to temp_ar
  Py_DECREF(output_ar);
  output_ar = temp_ar;
  /**
   * now we can call the solving routine. check self->solver to choose solver;
   * only values are "qr" for QR decomp or "svd" for singular value decomp.
   * each routine will set the coef_, intercept_, rank_, singular_, and fitted
   * members of the LinearRegression *self.
   */
  if (strcmp(self->solver, "qr") == 0) {
    // call QR solving routine (calls dgelsy). returns -1 on error
    if (qr_solver(self, input_ar, output_ar) < 0) {
      goto except;
    }
  }
  else if (strcmp(self->solver, "svd") == 0) {
    // call SVD solving routine (calls dgelss). returns -1 on error
    if (svd_solver(self, input_ar, output_ar) < 0) {
      goto except;
    }
  }
  // clean up input_ar, output_ar by Py_DECREF and return new ref to self
  Py_DECREF(input_ar);
  Py_DECREF(output_ar);
  Py_INCREF(self);
  return (PyObject *) self;
// clean up input_ar, output_ar and return NULL on exceptions
except:
  Py_XDECREF(input_ar);
  Py_XDECREF(output_ar);
  return NULL;
}

// docstring for the LinearRegression predict method
PyDoc_STRVAR(
  LinearRegression_predict_doc,
  "predict(X)"
  "\n--\n\n"
  "Compute predicted response given new inputs ``X``."
  "\n\n"
  "If the model has not been fitted, a :class:`RuntimeError` will be raised.\n"
  "Also, if ``X`` is not of type :class:`numpy.ndarray`, not C-contiguous,\n"
  "not memory-aligned, and dont' have ``dtype`` double."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "X : numpy.ndarray\n"
  "    Points to evaluate the model at, shape ``(n_samples, n_features)``"
  "\n\n"
  "Returns\n"
  "-------\n"
  "numpy.ndarray\n"
  "    Predicted response, shape ``(n_samples,)`` or ``(n_samples, n_targets)``"
);
/**
 * `predict` method for the `LinearRegression` class.
 * 
 * No keyword arguments needed, so `kwargs` is omitted.
 * 
 * @param self `LinearRegression *` instance
 * @param arg `PyObject *` representing the input matrix
 * @returns New reference, `PyArrayObject *` cast to `PyObject *` giving
 *    the responses predicted from the input.
 */
static PyObject *
LinearRegression_predict(LinearRegression *self, PyObject *arg)
{
  // if model is not fitted, raise RuntimeError
  if (!self->fitted) {
    PyErr_SetString(PyExc_RuntimeError, "cannot predict with unfitted model");
    return NULL;
  }
  /**
   * input ndarray, which we attempt to convert from arg. can use
   * NPY_ARRAY_IN_ARRAY instead of NPY_ARRAY_CARRAY since we don't need to
   * write to input_ar. NPY_DOUBLE type as usual.
   */
  PyArrayObject *input_ar = (PyArrayObject *) PyArray_FROM_OTF(
    arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  if (input_ar == NULL) {
    return NULL;
  }
  // check that input_ar has positive size and appropriate shape
  if (PyArray_SIZE(input_ar) < 1) {
    PyErr_SetString(PyExc_ValueError, "X must be nonempty");
    goto except_input_ar;
  }
  // check that input_ar and output_ar have appropriate shape
  if (PyArray_NDIM(input_ar) != 2) {
    PyErr_SetString(
      PyExc_ValueError, "X must have shape (n_samples, n_features)"
    );
    goto except_input_ar;
  }
  // for convenience, get number of rows and columns of input_ar
  npy_intp n_samples, n_features;
  n_samples = PyArray_DIM(input_ar, 0);
  n_features = PyArray_DIM(input_ar, 1);
/**
 * typically npy_intp has same size as long, 64-bit on 64-bit architecture.
 * if we aren't linked against Intel MKL (make links to ILP64, 64-bit int),
 * i.e. MKL_ILP64 not defind, and not linked against OpenBLAS built with
 * INTERFACE64=1 for 64-bit int, i.e. OPENBLAS_USE64BITINT not defined, we need
 * to check if the value of n_samples, n_features exceeds INT_MAX. if so,
 * raise OverflowError, else passing n_samples, n_features > 2^32 - 1 can lead
 * to weird results, possibly segmentation fault.
 */ 
#if !defined(MKL_ILP64) && !defined(OPENBLAS_USE64BITINT)
  if (n_samples > INT_MAX || n_features > INT_MAX) {
    PyErr_SetString(
      PyExc_OverflowError,
      "CBLAS/LAPACKE implementation does not support 64-bit indexing"
    );
    goto except_input_ar;
  }
#endif
  // output array, which will have NPY_ARRAY_CARRAY flags and type NPY_DOUBLE
  PyArrayObject *output_ar;
  // double * for input_ar, output_ar, and self->coef_ data
  double *input_data, *output_data, *coef_data;
  input_data = (double *) PyArray_DATA(input_ar);
  coef_data = (double *) PyArray_DATA((PyArrayObject *) self->coef_);
  // check shape of parameters. if single-target, i.e. PyArray_NDIM gives 1,
  // we use cblas_dgemv for coefficient multiplication.
  if (PyArray_NDIM((PyArrayObject *) self->coef_) == 1) {
    // if single-target, output_ar has shape (n_samples,)
    output_ar = (PyArrayObject *) PyArray_SimpleNew(
      1, PyArray_DIMS(input_ar), NPY_DOUBLE
    );
    if (output_ar == NULL) {
      goto except_input_ar;
    }
    // on success, set output_data. we need this later
    output_data = (double *) PyArray_DATA(output_ar);
    // get the double intercept from self->intercept_, a Python float. returns
    // -1 on failure so if PyErr_Occurred we have an exception
    double intercept = PyFloat_AsDouble(self->intercept_);
    if (PyErr_Occurred()) {
      goto except_output_ar;
    }
    // use cblas_dgemv to compute product input_ar @ self->coef_. the result is
    // written to the data pointer of output_ar.
    cblas_dgemv(
      CblasRowMajor, CblasNoTrans, (const MKL_INT) n_samples,
      (const MKL_INT) n_features, 1, (const double *) input_data,
      (const MKL_INT) n_features, (const double *) coef_data, 1, 0,
      (double *) output_data, 1
    );
    // add the intercept to each element of output_ar if nonzero
    if (intercept != 0) {
      for (npy_intp i = 0; i < n_samples; i++) {
        output_data[i] += intercept;
      }
    }
    // done computing our result, so return to main function body
  }
  // else if multi-target, i.e. PyArray_NDIM gives 2, we use cblas_dgemm
  else {
    // get number of targets (coef rows), i.e. PyArray_DIM(self->coef_, 0)
    npy_intp n_targets = PyArray_DIM((PyArrayObject *) self->coef_, 0);
    // create output_ar dims and output_ar, shape (n_samples, n_targets)
    npy_intp dims[] = {n_samples, n_targets};
    output_ar = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (output_ar == NULL) {
      goto except_input_ar;
    }
    // again, set output_data on success
    output_data = (double *) PyArray_DATA(output_ar);
    // use cblas_dgemm to compute product input_ar @ self->coef_.T, where the
    // results is written to the data pointer of output_ar
    cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasTrans, (const MKL_INT) n_samples,
      (const MKL_INT) n_targets, (const MKL_INT) n_features, 1,
      (const double *) input_data, (const MKL_INT) n_features,
      (const double *) coef_data, (const MKL_INT) n_features, 0,
      output_data, (const MKL_INT) n_targets
    );
    /**
     * in multi-target, if fit_intercept=False, self->intercept_ is Python
     * float with value 0. so if self->intercept_ not ndarray, it's just 0,
     * so we only need to handle intercept if self->intercept_ is ndarray.
     */
    if (PyArray_Check(self->intercept_)) {
      // get pointer to self->intercept_ data
      double *intercept_data = (double *) PyArray_DATA(
        (PyArrayObject *) self->intercept_
      );
      // add self->intercept_[j] to each jth column of output_ar
      for (npy_intp i = 0; i < n_samples; i++) {
        for (npy_intp j = 0; j < n_targets; j++) {
          output_data[i * n_targets + j] += intercept_data[j];
        }
      }
    }
    // done computing out result, so return to main function body
  }
  // clean up input_ar and return output_ar
  Py_DECREF(input_ar);
  return (PyObject *) output_ar;
// clean up input_ar, output_ar and return NULL on exceptions
except_output_ar:
  Py_DECREF(output_ar);
except_input_ar:
  Py_DECREF(input_ar);
  return NULL;
}

// methods of the LinearRegression type
static PyMethodDef LinearRegression_methods[] = {
  {
    "fit", (PyCFunction) LinearRegression_fit,
    METH_VARARGS, LinearRegression_fit_doc
  },
  {
    "predict", (PyCFunction) LinearRegression_predict,
    METH_O, LinearRegression_predict_doc
  },
  // sentinel marking end of array
  {NULL, NULL, 0, NULL}
};

// LinearRegression type docstring
PyDoc_STRVAR(
  LinearRegression_doc,
  "LinearRegression(*, fit_intercept=True, solver=\"svd\")"
  "\n--\n\n"
  "Ordinary least squares linear regression."
  "\n\n"
  "Directly calls into the LAPACKE routines ``dgelsy`` and ``dgelss``. Input\n"
  "matrix provided during fitting may be rank-deficient. Typically\n"
  "``solver=\"qr\"`` is faster than ``solver=\"svd\"``."
  "\n\n"
  "All members are read-only. Members listed under `Attributes`_ are\n"
  "available only after the model has been fitted. Note that when\n"
  "``solver=\"svd\"`` the value of ``singular_`` will differ from that of\n"
  "scikit-learn implementation, which also scales the columns to unit norm\n"
  "before calling a LAPACKE routine on the transformed data."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "fit_intercept : bool, default=True\n"
  "    Whether to compute the intercept for the model. If ``False``, then\n"
  "    the intercept will be set to zero.\n"
  "solver : {\"qr\", \"svd\"}, default=\"svd\"\n"
  "    Solution method for solving the resulting linear system after\n"
  "    factorization of the centered data matrix. ``\"qr\"`` results in a\n"
  "    call to ``dgelsy`` while ``\"svd\"`` results in a call to ``dgelss``."
  "\n\n"
  "Attributes\n"
  "----------\n"
  "coef_ : numpy.ndarray\n"
  "    Coefficients of the linear model, shape ``(n_features,)`` for\n"
  "    single output or ``(n_targets, n_features)`` for multioutput.\n"
  "intercept_ : float or numpy.ndarray\n"
  "    Intercept of the linear model. If multioutput, shape ``(n_targets,)``.\n"
  "rank_ : int\n"
  "    Effective rank of the centered input matrix.\n"
  "singular_ : numpy.ndarray\n"
  "    Singular values of the centered input matrix shape\n"
  "    ``(min(n_samples, n_features),)`` if ``solver=\"svd\"``, else ``None``."
  "\n\n"
  "Methods\n"
  "-------"
);
// the LinearRegression type object
static PyTypeObject LinearRegression_type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  // module-qualified name, docstring, size of LinearRegression struct
  .tp_name = "_linreg.LinearRegression",
  .tp_doc = LinearRegression_doc,
  .tp_basicsize = sizeof(LinearRegression),
  // tp_itemsize only nonzero when object has notion of "size", ex. a list type
  .tp_itemsize = 0,
  // adding Py_TPFAGS_BASETYPE allows LinearRegression to be subclassed
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  // __new__, __init__, deallocator of LinearRegression class
  .tp_new = LinearRegression_new,
  .tp_init = (initproc) LinearRegression_init,
  .tp_dealloc = (destructor) LinearRegression_dealloc,
  // members, getset members, and methods of the LinearRegression class
  .tp_members = LinearRegression_members,
  .tp_getset = LinearRegression_getsets,
  .tp_methods = LinearRegression_methods
};

// _linreg methods, possibly including EXTERNAL_* wrappers
static PyMethodDef _linreg_methods[] = {
// if EXPOSE_INTERNAL is defined, we make the EXPOSED_* methods accessible.
// again, defined(__INTELLISENSE__) lets VS Code Intellisense work here
#if defined(__INTELLISENSE__) || defined(EXPOSE_INTERNAL)
  {
    "EXPOSED_npy_vector_matrix_mean",
    (PyCFunction) EXPOSED_npy_vector_matrix_mean,
    METH_O,
    EXPOSED_npy_vector_matrix_mean_doc
  },
  {
    "EXPOSED_compute_intercept",
    (PyCFunction) EXPOSED_compute_intercept,
    METH_VARARGS,
    EXPOSED_compute_intercept_doc
  },
#endif /* EXPOSE_INTERNAL */
  // sentinel marking end of array
  {NULL, NULL, 0, NULL}
};

// _linreg module docstring
PyDoc_STRVAR(
  _linreg_doc,
  "C implementation of OLS linear regression solved by QR or SVD."
  "\n\n"
  "Provides a linear regression estimator with scikit-learn like interface.\n"
  "Fitting method directly calls into LAPACKE routines ``dgelsy``, ``dgelss``."
);
// _linreg module definition
static PyModuleDef _linreg_module = {
  PyModuleDef_HEAD_INIT,
  // name, docstring, size = -1 to disable subinterpreter support, methods
  .m_name = "_linreg",
  .m_doc = _linreg_doc,
  .m_size = -1,
  .m_methods = _linreg_methods
};

// module initialization function
PyMODINIT_FUNC
PyInit__linreg(void)
{
  // import NumPy Array C API. automatically returns NULL on error.
  import_array();
  // check if LinearRegression_type is ready. NULL on error
  if (PyType_Ready(&LinearRegression_type) < 0) {
    return NULL;
  }
  // create module. NULL on error
  PyObject *this_mod = PyModule_Create(&_linreg_module);
  if (this_mod == NULL) {
    return NULL;
  }
  // add LinearRegression_type to module. note that reference is stolen only
  // upon success, so we Py_INCREF first and Py_DECREF on failure.
  Py_INCREF(&LinearRegression_type);
  if (
    PyModule_AddObject(
      this_mod, "LinearRegression", (PyObject *) &LinearRegression_type
    ) < 0
  ) {
    Py_DECREF(&LinearRegression_type);
    Py_DECREF(this_mod);
    return NULL;
  }
  return this_mod;
}