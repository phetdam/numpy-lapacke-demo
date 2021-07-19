/**
 * @file _linreg.c
 * @author Derek Huang <djh458@stern.nyu.edu>
 * @brief C implementation of OLS linear regression solved by QR or SVD.
 * 
 * Directly calls into LAPACKE routines dgelsy, dgelss to solve.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <limits.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

// npt_lapacke_demo/*.h automatically handles the different includes
// depending on whether Intel MKL, OpenBLAS, or system CBLAS/LAPACKE is linked
#include "npy_lapacke_demo/cblas.h"
#include "npy_lapacke_demo/lapacke.h"

// make available macros defined in linreginternal.h for API initialization
#define LINREG_MODULE
#include "linreginternal.h"

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
static const char *LinearRegression_kwargs[] = {
  "fit_intercept", "solver", NULL
};
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
      args, kwargs, "|$ps", (char **) LinearRegression_kwargs,
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

/**
 * `__repr__` method for the `LinearRegression` class.
 * 
 * @param self `LinearRegression *` to instance returned by `__init__`
 * @returns New reference to `PyUnicodeObject *` on success giving string
 *     representation for the instance, `NULL` on error with exception set.
 */
static PyObject *
LinearRegression_repr(LinearRegression *self)
{
  return PyUnicode_FromFormat(
    "LinearRegression(*, fit_intercept=%s, solver=%s)",
    (self->fit_intercept) ? "True" : "False", self->solver
  );
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
  "not memory-aligned, and doesn't have ``dtype`` double, it will be copied."
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

/**
 * Computes the weighted univariate R^2 given true and predicted responses.
 * 
 * Do NOT call without proper argument checking.
 * 
 * Able to handle cases where the elements to use in calculation are not
 * contiguous but evenly spaced in memory, for example if we are computing the
 * R^2 for each column of a matrix laid out in row-major.
 * 
 * Always returns 1 if true values are equal to predicted values, even if the
 * true values are the same exact value, i.e. variance of true is zero.
 * 
 * @param y_true `const double *` array holding the true response values. Must
 *     have length at least `n_samples * ldim`. If representing a matrix, must
 *     represent the data of the matrix in row-major layout.
 * @param y_pred `const double *` array holding the predicted response values.
 *     Must have length at least `n_samples * ldim`. If representing a matrix,
 *     must represent the data of the matrix in row-major layout.
 * @param weights `const double *` array of at least `n_samples` giving the
 *     nonnegative weights for the samples used in computing R^2. If `NULL`,
 *     then all the weights are set to 1 to compute an unweighted R^2. Weights
 *     do not have to add up to 1; they are used as-is.
 * @param n_samples `npy_intp` indicating how many samples to use when
 *     computing the univariate R^2. Should be at least `1`, however.
 * @param ldim `npy_intp` indicating the leading dimension of `y_true`,
 *     `y_pred`. If `y_true`, `y_pred` are flat arrays, set to `1`, else if
 *     `y_true`, `y_pred` are matrices in row-major layout, set to the number
 *     of elements in a row. Same as `lda` in many BLAS/LAPACK functions.
 * @returns Univariate R^2
 */
static double
weighted_univariate_r2(
  const double *y_true, const double *y_pred, const double *weights,
  npy_intp n_samples, npy_intp ldim
)
{
  // sample mean of y_true, weighted sample variance of y_true * weight_sum,
  // weighted sum of squared differences between y_true and y_pred, weight sum
  double y_true_mean, y_true_uvar, pred_uvar, weight_sum;
  y_true_mean = y_true_uvar = pred_uvar = 0;
  // if weights is NULL, weight_sum is n_samples, else compute sum of weights
  if (weights == NULL) {
    weight_sum = n_samples;
  }
  else {
    weight_sum = 0;
    for (npy_intp i = 0; i < n_samples; i++) {
      weight_sum += weights[i];
    }
    weight_sum /= n_samples;
  }
  /**
   * compute the weighted mean of of the true responses. note we are indexing
   * with 1 since it makes it easier to use ldim to skip elements this way.
   * note that when weights == NULL, i.e. no weights, cur_weight already 1.
   */
  for (npy_intp i = 1; i <= n_samples; i++) {
    y_true_mean += ((weights == NULL) ? 1 : weights[i]) * y_true[ldim * i - 1];
  }
  y_true_mean /= weight_sum;
  // compute the weighted sample variance of y_true * n_samples
  for (npy_intp i = 1; i <= n_samples; i++) {
    y_true_uvar += ((weights == NULL) ? 1 : weights[i]) * pow(
      y_true[ldim * i - 1] - y_true_mean, 2
    );
  }
  /**
   * compute the weighted sum of squared differences between y_true, y_pred,
   * i.e. the "residual sum of squares", stored in pred_uvar. this quantity can
   * also be interpreted as the weighted sample variance of y_true conditional
   * on input points and parameters multiplied by n_samples.
   */
  for (npy_intp i = 0; i <= n_samples; i++) {
    pred_uvar += ((weights == NULL) ? 1 : weights[i]) * pow(
      y_true[ldim * i - 1] - y_pred[ldim * i - 1], 2
    );
  }
  // if sum of squared differences is zero, return 1. note y_true_uvar might be
  // 0, but we ignore this since the model is technically "perfect".
  if (pred_uvar == 0) {
    return 1;
  }
  // if y_true_uvar is zero, then return -NPY_NAN (edge case)
  if (y_true_uvar == 0) {
    return (double) -NPY_NAN;
  }
  // else return the (weighted) univariate R^2
  return 1 - pred_uvar / y_true_uvar;
}

// docstring for the LinearRegression score method
PyDoc_STRVAR(
  LinearRegression_score_doc,
  "score(X, y, sample_weight=None, multioutput=\"uniform_average\")"
  "\n--\n\n"
  "Return coefficient of determination :math:`R^2` of the predictions."
  "\n\n"
  "If the model has not been fitted, a :class:`RuntimeError` will be raised.\n"
  "Note that ``X``, ``y`` will be copied if they are not of type\n"
  ":class:`numpy.ndarray`, not C-contiguous, not memory-aligned, or don't\n"
  "have ``dtype`` double. This function provides a similar implementation to\n"
  "the scikit-learn ``r2_score`` function with fewer multi-output options."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "X : numpy.ndarray\n"
  "    Points to evaluate the model at, shape ``(n_samples, n_features)``\n"
  "y : numpy.ndarray\n"
  "    True response values for the value, shape ``(n_samples, n_targets)``\n"
  "sample_weight : numpy.ndarray, default=None\n"
  "    Sample weights for each of the data points.\n"
  "multioutput : {\"uniform_average\", \"raw_values\"}, "
  "default=\"uniform_average\"\n"
  "    Defines how :math:`R^2` scores for multiple outputs should be\n"
  "    aggregated. If \"uniform_average\", the unweighted mean of the scores\n"
  "    is returned, while if \"raw_values\", a numpy.ndarray of the\n"
  "    individual scores will be returned, shape ``(n_targets,)``."
  "\n\n"
  "Returns\n"
  "-------\n"
  "float\n"
  "    Coefficient of determination for the "
);
// argument names known to LinearRegression_score
static const char *LinearRegression_score_argnames[] = {
  "X", "y", "sample_weight", "multioutput", NULL
};
/**
 * `score` method for the `LinearRegression` class.
 * 
 * Implementation is based off of the `r2_score` function in `sklearn.metrics`.
 * 
 * @param self `LinearRegression *` instance
 * @param args `PyObject *` positional args tuple
 * @param kwargs `PyObject *` keyword args dict, may be `NULL`
 * @returns New reference to either `PyFloatObject *` or `PyArrayObject *` else
 *     `NULL` on error with exception set.
 */
static PyObject *
LinearRegression_score(LinearRegression *self, PyObject *args, PyObject *kwargs)
{
  /**
   * input matrix, response matrix/vector, predicted response, sample weights,
   * array of R^2 scores that exists in multioutput case. weights and res_ar
   * NULL by default, so they must only be Py_XDECREF'd.
   */
  PyArrayObject *X, *y_true, *y_pred, *weights, *res_ar;
  weights = res_ar = NULL;
  // how to treat scoring in multioutput case. default "uniform_average"
  const char *multioutput;
  multioutput = "uniform_average";
  // number of samples, features, targets
  npy_intp n_samples, n_features, n_targets;
  // returned score(s); might be PyFloatObject * or PyArrayObject *
  PyObject *res;
  // holds current R2 score, data pointers for y_true, y_pred
  double r2_score, *y_true_data, *y_pred_data;
  // if model is not fitted, raise RuntimeError
  if (!self->fitted) {
    PyErr_SetString(PyExc_RuntimeError, "cannot score with unfitted model");
    return NULL;
  }
  // else fitted, so we can score. parse args and kwargs
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "OO|Os", (char **) LinearRegression_score_argnames,
      &X, &y_true, &weights, &multioutput
    )
  ) {
    return NULL;
  }
  // since this check is easy, we do this first. check that value of the
  // multioutput const char * is one of the two accepted values
  if (
    strcmp(multioutput, "uniform_average") != 0 &&
    strcmp(multioutput, "raw_values") != 0
  ) {
    PyErr_SetString(
      PyExc_ValueError,
      "multioutput must be one of (\"uniform_average\", \"raw_values\")"
    );
    return NULL;
  }
  // convert X, y to NPY_DOUBLE, NPY_ARRAY_IN_ARRAY ndarrays. drop borrowed refs
  X = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) X, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  if (X == NULL) {
    return NULL;
  }
  y_true = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) y_true, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
  );
  if (y_true == NULL) {
    goto except_X;
  }
  // convert weights only if weights is not NULL. discard borrowed ref.
  if (weights != NULL) {
    weights = (PyArrayObject *) PyArray_FROM_OTF(
      (PyObject *) weights, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    if (weights == NULL) {
      goto except_y_true;
    }
  }
  // check dimensions. all must be nonempty. note weights might be NULL
  if (PyArray_SIZE(X) == 0) {
    PyErr_SetString(PyExc_ValueError, "X must be nonempty");
    goto except_weights;
  }
  if (PyArray_SIZE(y_true) == 0) {
    PyErr_SetString(PyExc_ValueError, "y must be nonempty");
    goto except_weights;
  }
  if (weights != NULL && PyArray_SIZE(weights) == 0) {
    PyErr_SetString(PyExc_ValueError, "weights must be nonempty");
    goto except_weights;
  }
  // X must be 2D, y must be 1D or 2D, weights must be 1D
  if (PyArray_NDIM(X) != 2) {
    PyErr_SetString(PyExc_ValueError, "X must be 2D");
    goto except_weights;
  }
  if (PyArray_NDIM(y_true) != 1 && PyArray_NDIM(y_true) != 2) {
    PyErr_SetString(PyExc_ValueError, "y must be 1D or 2D");
    goto except_weights;
  }
  if (weights != NULL && PyArray_NDIM(weights) != 1) {
    PyErr_SetString(PyExc_ValueError, "weights must be 1D");
    goto except_weights;
  }
  // use X to get n_samples, n_features. if n_samples < 2, return NaN
  n_features = PyArray_DIM(X, 0);
  n_samples = PyArray_DIM(X, 1);
  if (n_samples < 2) {
    // if warning turned into exception by user code, use normal cleanup goto
    if (
      PyErr_WarnEx(
        PyExc_UserWarning, "R^2 score is not well-defined with < 2 samples", 1
      ) < 0
    ) {
      goto except_weights;
    }
    // else do manual cleanup. note NULL may be returned on error.
    Py_XDECREF(weights);
    Py_DECREF(y_true);
    Py_DECREF(X);
    return PyFloat_FromDouble((double) NPY_NAN);
  }
  // check that y, weights have first dimension equal to n_samples
  if (PyArray_DIM(y_true, 0) != n_features) {
    PyErr_SetString(
      PyExc_ValueError, "y must have shape (n_samples, n_targets)"
    );
    goto except_weights;
  }
  if (weights != NULL && PyArray_SIZE(weights) != n_features) {
    PyErr_SetString(PyExc_ValueError, "weights must have shape (n_samples,)");
    goto except_weights;
  }
  // same number of samples so use y_true to get n_targets
  n_targets = (PyArray_NDIM(y_true) == 1) ? 1 : PyArray_DIM(y_true, 1);
  // call LinearRegression_predict to get predicted y values. also NPY_DOUBLE
  // type with NPY_ARRAY_CARRAY flags (writable)
  y_pred = (PyArrayObject *) LinearRegression_predict(self, (PyObject *) X);
  if (y_pred == NULL) {
    goto except_weights;
  }
  // get pointers to the data of y_true, y_pred
  y_true_data = (double *) PyArray_DATA(y_true);
  y_pred_data = (double *) PyArray_DATA(y_pred);
  // handle single and multi-target cases separately
  if(n_targets == 1) {
    // compute univariate R^2 and create Python float from r2_score
    r2_score = weighted_univariate_r2(
      y_true_data, y_pred_data,
      (weights == NULL) ? NULL : (double *) PyArray_DATA(weights), n_samples, 1
    );
    res = PyFloat_FromDouble(r2_score);
  }
  else {
    // allocate new NPY_DOUBLE, NPY_ARRAY_CARRAY ndarray for R^2 scores. note
    // we borrow the dims of y_true for this and use some pointer arithmetic
    res_ar = (PyArrayObject *) PyArray_SimpleNew(
      1, PyArray_DIMS(y_true) + 1, NPY_DOUBLE
    );
    if (res_ar == NULL) {
      goto except_y_pred;
    }
    // pointer to data of res since it needs to be filled with values
    double *res_data;
    res_data = (double *) PyArray_DATA(res_ar);
    // compute R^2 for each column of y_true, y_pred; i.e. ldim = n_targets
    // and the starting pointer to the data is shifted by the column index
    for (npy_intp i = 0; i < n_targets; i++) {
      res_data[i] = weighted_univariate_r2(
        y_true_data + i, y_pred_data + i,
        (weights == NULL) ? NULL : (double *) PyArray_DATA(weights),
        n_samples, n_targets
      );
    }
    // if multioutput is "uniform_average", set res to the average of the
    // values in res_data, i.e. the unweighted average of each column R^2 
    if (strcmp(multioutput, "uniform_average") == 0) {
      r2_score = 0;
      for (npy_intp i = 0; i < n_targets; i++) {
        r2_score += res_data[i];
      }
      r2_score /= n_targets;
      res = PyFloat_FromDouble(r2_score);
    }
    // else just set res to res_ar, i.e. multioutput is "raw_values". since
    // res_ar will be Py_DECREF'd later, Py_INCREF res
    else {
      res = (PyObject *) res_ar;
      Py_INCREF(res);
    }
  }
  // if res is NULL, exception, so clean up. since res_ar is Py_XDECREF'd, this
  // works just fine regardless of whether or not res_ar points to memory.
  if (res == NULL) {
    goto except_res_ar;
  }
  // else clean up and return
  Py_XDECREF(res_ar);
  Py_DECREF(y_pred);
  Py_XDECREF(weights);
  Py_DECREF(y_true);
  Py_DECREF(X);
  return res;
// clean up on error
except_res_ar:
  Py_XDECREF(res_ar);
except_y_pred:
  Py_DECREF(y_pred);
except_weights:
  Py_XDECREF(weights);
except_y_true:
  Py_DECREF(y_true);
except_X:
  Py_DECREF(X);
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
  {
    "score", (PyCFunction) LinearRegression_score,
    METH_VARARGS | METH_KEYWORDS, LinearRegression_score_doc
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
  .tp_methods = LinearRegression_methods,
  // __repr__ method for the LinearRegression class
  .tp_repr = (reprfunc) LinearRegression_repr
};

// _linreg module docstring
PyDoc_STRVAR(
  _linreg_doc,
  "C implementation of OLS linear regression solved by QR or SVD."
  "\n\n"
  "Provides a linear regression estimator with scikit-learn like interface.\n"
  "Fitting method directly calls into LAPACKE routines ``dgelsy``, ``dgelss``."
);
// _linreg module definition. no available functions
static PyModuleDef _linreg_module = {
  PyModuleDef_HEAD_INIT,
  // name, docstring, size = -1 to disable subinterpreter support, methods
  .m_name = "_linreg",
  .m_doc = _linreg_doc,
  .m_size = -1,
  .m_methods = NULL
};

// module initialization function
PyMODINIT_FUNC
PyInit__linreg(void)
{
  // PyObject * for module and capsule, void * array for C API (static!)
  PyObject *module, *c_api_obj;
  static void *Py__linreg_API[Py__linreg_API_pointers];
  // import NumPy Array C API. automatically returns NULL on error.
  import_array();
  // check if LinearRegression_type is ready. NULL on error
  if (PyType_Ready(&LinearRegression_type) < 0) {
    return NULL;
  }
  // create module. NULL on error
  module = PyModule_Create(&_linreg_module);
  if (module == NULL) {
    return NULL;
  }
  // add LinearRegression_type to module. note that reference is stolen only
  // upon success, so we Py_INCREF first and Py_DECREF on failure.
  Py_INCREF(&LinearRegression_type);
  if (
    PyModule_AddObject(
      module, "LinearRegression", (PyObject *) &LinearRegression_type
    ) < 0
  ) {
    Py_DECREF(&LinearRegression_type);
    goto except_module;
  }
  // initialize the pointers for the C function pointer API
  Py__linreg_API[Py__linreg_npy_vector_matrix_mean_NUM] = \
    (void *) npy_vector_matrix_mean;
  Py__linreg_API[Py__linreg_compute_intercept_NUM] = \
    (void *) compute_intercept;
  Py__linreg_API[Py__linreg_weighted_univariate_r2_NUM] = \
    (void *) weighted_univariate_r2;
  /**
   * create capsule containing address to C array API. on error, must XDECREF
   * c_api_obj, as it may be NULL on error. &LinearRegression_type reference
   * has been previously stolen, so no Py_DECREF of it on error.
   */
  c_api_obj = PyCapsule_New(
    (void *) Py__linreg_API, "npy_lapacke_demo.regression._linreg._C_API", NULL
  );
  // PyModule_AddObject returns NULL + sets exception if value arg is NULL
  if (PyModule_AddObject(module, "_C_API", c_api_obj) < 0) {
    Py_XDECREF(c_api_obj);
    goto except_module;
  }
  return module;
// clean up module on exception
except_module:
  Py_DECREF(module);
  return NULL;
}