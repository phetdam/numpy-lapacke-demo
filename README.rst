.. README.rst for npy_openblas_demo

npy_lapacke_demo
================

A tiny Python package demonstrating how to use `LAPACKE`__ and `CBLAS`__ with
NumPy arrays in C extension modules.

.. __: https://www.netlib.org/lapack/lapacke.html

.. __: http://www.netlib.org/blas/


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