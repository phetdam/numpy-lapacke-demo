.. README.rst for npy_lapacke_demo

npy_lapacke_demo
================

A small Python package demonstrating how to use `LAPACKE`__ and `CBLAS`__ with
NumPy arrays in C extension modules.

Also includes an example of making LAPACKE calls from normal C code in the
`lapacke_demo`__ directory. See the corresponding ``lapacke_demo/README.rst``
for details on building and running the example.

.. __: https://www.netlib.org/lapack/lapacke.html

.. __: http://www.netlib.org/blas/

.. __: https://github.com/phetdam/npy_lapacke_demo/tree/master/lapacke_demo


Installation
------------

From source
~~~~~~~~~~~

   Important Note:

   Linking against Intel MKL has proven to be rather difficult. The linker line
   suggested by the Intel Link Line Advisor leads to fatal errors during
   runtime since some symbols cannot be found, while using the single dynamic
   library ``libmkl_rt.so`` appears to work only when the environment variable
   ``MKL_INTERFACE_LAYER`` is set to ``GNU,ILP64`` or ``GNU,LP64``. Other 
   values result in segfaults whenever Intel MKL functions are called. For
   further details on setting the Intel MKL interface and threading layer,
   see the `dedicated Intel article`__.

   .. __: https://software.intel.com/content/www/us/en/develop/documentation/
      onemkl-linux-developer-guide/top/linking-your-application-with-the-intel-
      oneapi-math-kernel-library/linking-in-detail/dynamically-selecting-the-
      interface-and-threading-layer.html

TBD. However, the code will be written so that the C extensions can be built
with either `Intel MKL`__, `OpenBLAS`__, or standard system CBLAS and LAPACKE
implementations. But note that unless linked against Intel MKL using ILP64
interface [#]_ or with OpenBLAS using 64-bit ``int``, i.e. built with
``INTERFACE64=1``, no ``npy_lapacke_demo`` function or method that accepts
NumPy arrays should be passed arrays containing more elements than can be held
in a 32-bit ``int``, as an ``OverflowError`` will be raised. However, since
this package is intended for demonstration purposes, you shouldn't be using it
for true "big data" applications anyways.

.. __: https://software.intel.com/content/www/us/en/develop/documentation/
   onemkl-developer-reference-c/top.html

.. __: https://www.openblas.net/

.. [#] The ILP64 interface for Intel MKL uses 64-bit integers. See the
   `Intel article on ILP64 vs. LP64`__ for more details.

.. __: https://software.intel.com/content/www/us/en/develop/documentation/
   onemkl-linux-developer-guide/top/linking-your-application-with-the-intel-
   oneapi-math-kernel-library/linking-in-detail/linking-with-interface-
   libraries/using-the-ilp64-interface-vs-lp64-interface.html


From PyPI
~~~~~~~~~

TBD. `manylinux2010`__ and Windows wheels using OpenBLAS are planned.

.. __: https://github.com/pypa/manylinux


Package contents
----------------

The ``npy_lapacke_demo`` package contains the ``regression`` and ``solvers``
subpackages. The ``regression`` subpackage provides the ``LinearRegression``
class, implemented like a `scikit-learn`__ estimator, which can be used to fit
a linear model with optional intercept by ordinary least squares, using either
QR or singular value decompositions. The ``solvers`` subpackage provides the
``mnewton`` local minimizer, implemented such that it may be used as a frontend
for `scipy.optimize.minimize`__ by passing ``mnewton`` to the ``method``
keyword argument. ``mnewton`` implements Newton's method with a Hessian
modification, where any Hessians that are not positive definite have a multiple
of the identity added to them to make the Newton direction using the modified
Hessian a descent direction. ``mnewton`` implements Algorithm 3.3 on page 51 of
Nocedal and Wright's *Numerical Optimization* and uses the returned lower
Cholesky factor of the modified Hessian when computing the descent direction.
The step size for the line search satisfies the Armijo condition and is chosen
using a backtracking line search.

.. __: https://scikit-learn.org/stable/index.html

.. __: https://docs.scipy.org/doc/scipy/reference/generated/
   scipy.optimize.minimize.html


Code examples
-------------

Here are some examples that demonstrate how the ``LinearRegression`` class and
``mnewton`` minimizer could be used. To run the first example, scikit-learn
must be installed. NumPy and SciPy must of course be installed.

``LinearRegression``
~~~~~~~~~~~~~~~~~~~~

Fit a linear model with intercept by ordinary least squares on the Boston house
prices data using SVD.

.. code:: python3

   from sklearn.datasets import load_boston
   from sklearn.metrics import r2_score
   from sklearn.model_selection import train_test_split

   from npy_lapacke_demo.regression import LinearRegression

   X, y = load_boston(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
   lr = LinearRegression(solver="svd").fit(X_train, y_train)
   # no implementation of the score method, so we use predictions with r2_score
   print(f"test R2: {r2_score(y_test, lr.predict(X_test))}")

``mnewton``
~~~~~~~~~~~

Minimize the multivariate Rosenbrock function using Newton's method with
diagonal Hessian modification.

.. code:: python3

   import numpy as np
   from scipy.optimize import rosen, rosen_der, rosen_hess

   from npy_lapacke_demo.solvers import mnewton

   res = mnewton(rosen, np.zeros(5), jac=rosen_der, hess=rosen_hess)
   print(res)