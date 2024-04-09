.. README.rst for numpy-lapacke-demo

numpy-lapacke-demo
==================

.. image:: https://img.shields.io/github/workflow/status/phetdam/
   numpy-lapacke-demo/build?logo=github
   :target: https://github.com/phetdam/numpy-lapacke-demo/actions
   :alt: GitHub Workflow Status

A small Python package demonstrating how to use `LAPACKE`__ and `CBLAS`__ with
NumPy arrays in C extension modules.

The project repository also includes an example of making LAPACKE calls from
normal C code in the `lapacke_demo`__ directory. See the corresponding
``lapacke_demo/README.rst`` for details on building and running the example.

.. __: https://www.netlib.org/lapack/lapacke.html

.. __: http://www.netlib.org/blas/

.. __: https://github.com/phetdam/numpy-lapacke-demo/tree/master/lapacke_demo


Installation
------------

From source
~~~~~~~~~~~

   Important Note:

   Although the code has been written to allow linking against Intel MKL, in
   practice this has not proven possible, as calls to Intel MKL routines result
   in segmentation faults. The linker line suggested by the Intel Link Line
   Advisor leads to fatal errors during runtime since some symbols cannot be
   found, while using the single dynamic library ``libmkl_rt.so`` appears to
   [occasionally] not segfault when ``MKL_INTERFACE_LAYER`` is set to
   ``GNU,ILP64`` or ``GNU,LP64``. The `dedicated Intel article`__ gives further
   details on setting the Intel MKL interface and threading layer.

.. __: https://software.intel.com/content/www/us/en/develop/documentation/
   onemkl-linux-developer-guide/top/linking-your-application-with-the-intel-
   oneapi-math-kernel-library/linking-in-detail/dynamically-selecting-the-
   interface-and-threading-layer.html

TBD. However, the C extension modules can be built with either `OpenBLAS`__ or
standard system CBLAS and LAPACKE implementations. But note that unless linked
against Intel MKL using ILP64 interface [#]_ or with OpenBLAS using 64-bit
``int``, i.e. built with ``INTERFACE64=1``, no ``npypacke`` function or method
that accepts NumPy arrays should be passed arrays requiring 64-bit indexing as
an ``OverflowError`` will be raised. Do note that this package is for
demonstration purposes and so shouldn't be used for true "big data"
applications anyways.

.. __: https://www.openblas.net/

.. [#] The ILP64 interface for Intel MKL uses 64-bit integers. See the
   `Intel article on ILP64 vs. LP64`__ for more details.

.. __: https://software.intel.com/content/www/us/en/develop/documentation/
   onemkl-linux-developer-guide/top/linking-your-application-with-the-intel-
   oneapi-math-kernel-library/linking-in-detail/linking-with-interface-
   libraries/using-the-ilp64-interface-vs-lp64-interface.html


From PyPI
~~~~~~~~~

TBD. Local x86-64 builds on WSL Ubuntu 20.04 LTS linking against OpenBLAS
0.3.17 or against ``libblas`` 3.9.0 and ``liblapacke`` 3.9.0 installed using
``apt`` have succeeded, as have builds on GitHub Actions runners for
`manylinux1`__, manylinux2010, and MacOS linked against the latest OpenBLAS
(0.3.17), built using `cibuildwheel`__. However, there have been issues with
building wheels for Windows with the latest `OpenBLAS Windows binaries`__.

.. __: https://github.com/pypa/manylinux

.. __: https://cibuildwheel.readthedocs.io/

.. __: https://github.com/xianyi/OpenBLAS/releases


Package contents
----------------

The ``npypacke`` package contains the ``regression`` and ``solvers``
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

Fit a linear model with intercept by least squares on the Boston house prices
data using QR decomposition.

.. code:: python

   from sklearn.datasets import load_boston
   from sklearn.model_selection import train_test_split

   from npypacke.regression import LinearRegression

   X, y = load_boston(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
   lr = LinearRegression(solver="qr").fit(X_train, y_train)
   print(lr.score(X_test, y_test))

``mnewton``
~~~~~~~~~~~

Minimize the multivariate Rosenbrock function using Newton's method with
diagonal Hessian modification.

.. code:: python

   import numpy as np
   from scipy.optimize import rosen, rosen_der, rosen_hess

   from npypacke.solvers import mnewton

   res = mnewton(rosen, np.zeros(5), jac=rosen_der, hess=rosen_hess)
   print(res.x)
