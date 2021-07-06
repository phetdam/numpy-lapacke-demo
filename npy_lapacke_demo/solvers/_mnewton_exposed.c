/**
 * @file _mnewton__exposed.c
 * @author Derek Huang <djh458@stern.nyu.edu>
 * @brief Wrappers for internal C functions in `_mnewton.c`.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// gives access to internal C functions in _mnewton.c
#include "mnewton.h"

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

// _mnewton_exposed methods (wrap internal functions in _mnewton)
static PyMethodDef _mnewton_exposed_methods[] = {
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
#if 0
  {
    "tuple_prepend_single",
    (PyCFunction) tuple_prepend_single,
    METH_VARARGS | METH_KEYWORDS, tuple_prepend_single_doc
  },
  {
    "populate_OptimizeResult",
    (PyCFunction) populate_OptimizeResult,
    METH_VARARGS | METH_KEYWORDS, populate_OptimizeResult_doc
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
#endif
  // sentinel marking end of array
  {NULL, NULL, 0, NULL}
};

// _mnewton_exposed module definition
static PyModuleDef _mnewton_exposed_module = {
  PyModuleDef_HEAD_INIT,
  // name, docstring, size = -1 to disable subinterpreter support, methods
  .m_name = "_mnewton_exposed",
  .m_doc = "Wrappers for unit testing internal C functions in _mnewton.",
  .m_size = -1,
  .m_methods = _mnewton_exposed_methods
};

// module initialization function
PyMODINIT_FUNC
PyInit__mnewton_exposed(void)
{
  // import NumPy Array C API. automatically returns NULL on error.
  import_array();
  // import _mnewton C API.  automatically returns NULL on error.
  import__mnewton();
  // create module and return. NULL on error
  return PyModule_Create(&_mnewton_exposed_module);
}