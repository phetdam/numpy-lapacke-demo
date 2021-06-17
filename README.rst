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

TBD. However, the code will be written so that the C extensions can be built
with either `Intel MKL`__, `OpenBLAS`__, or standard system CBLAS and LAPACKE
implementations.

.. __: https://software.intel.com/content/www/us/en/develop/documentation/
   onemkl-developer-reference-c/top.html

.. __: https://www.openblas.net/


From PyPI
~~~~~~~~~

TBD. `manylinux2010`__ and Windows wheels using OpenBLAS are planned.

.. __: https://github.com/pypa/manylinux