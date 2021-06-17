.. README.rst for npy_openblas_demo

npy_openblas_demo
=================

A tiny Python package demonstrating how to use OpenBLAS [#]_ with NumPy
arrays in C extension modules.

.. [#] The package name is a misnomer, as the package may be linked against a
   few different `CBLAS`_ and `LAPACKE`_ implementations. See the
   `Installation`_ section for more details.

.. __: http://www.netlib.org/blas/

.. __: https://www.netlib.org/lapack/lapacke.html


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