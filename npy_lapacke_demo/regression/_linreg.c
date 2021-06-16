/**
 * @file _linreg.c
 * @brief C implementation of OLS linear regression solved by QR or SVD.
 * 
 * Directly calls into LAPACKE routines dgelsy, dgelss.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// npt_lapacke_demo/lapacke.h automatically handles the different includes
// depending on whether Intel MKL or OpenBLAS/LAPACKE is linked
#include "npy_lapacke_demo/lapacke.h"

// struct representing our linear regression estimator
typedef struct {
  PyObject_HEAD
  // 1 if model will fit the intercept, 0 otherwise
  char fit_intercept;
  // name of the solution method
  const char *solver;
  // coefficients and intercept of the linear model
  PyObject *coef_, *intercept_;
  // effective rank and singular values of the data matrix. singular_ is NULL
  // unless we solve using SVD. trailing underscore follows sklearn convention.
  int rank_;
  PyObject *singular_;
  // private attribute. 1 when model is fitted, 0 otherwise. the members with
  // trailing underscores are accessible iff fitted is 1.
  char fitted;
} LinearRegression;

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
  self->fit_intercept = 0;
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
    PyErr_SetString(PyExc_AttributeError, "coef_ only availble after fitting");
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
  "not memory aligned, and don't have ``dtype`` double."
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
 * @returns `PyObject *` to `self` to allow method chaining.
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
    // TODO: call QR solver, giving self, input_ar, output_ar
  }
  else if (strcmp(self->solver, "svd") == 0) {
    // TODO: call SVD solver, giving self, input_ar, output_ar
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

// methods of the LinearRegression type
static PyMethodDef LinearRegression_methods[] = {
  {
    "fit", (PyCFunction) LinearRegression_fit,
    METH_VARARGS, LinearRegression_fit_doc
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
  "available only after the model has been fitted."
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
  "    Effective rank of the input matrix.\n"
  "singular_ : numpy.ndarray\n"
  "    Singular values of the input matrix shape\n"
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
  .m_name = "_linreg",
  .m_doc = _linreg_doc,
  .m_size = -1
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