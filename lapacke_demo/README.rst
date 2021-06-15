lapacke_demo
============

Example code for dynamically linking against OpenBLAS, LAPACKE, or Intel MKL
on a Linux system.

The code initializes a small matrix with shape ``(4, 3)``, stored in row-major
as a ``double`` array, and performs a QR decomposition on the array using
`LAPACKE_dgeqrf`__ and `LAPACKE_dorgqr`__. If linking to a version of the
`GNU Scientific Library`__ (GSL), a random matrix with shape ``(4, 3)``,
elements drawn from the standard normal distribution, will then be printed to
screen. The PRNG used to generate the elements is seeded, so running
``lapacke_demo`` executable multiple times results in the same output.

.. __: https://software.intel.com/content/www/us/en/develop/documentation/
   onemkl-developer-reference-c/top/lapack-routines/lapack-least-squares-and-
   eigenvalue-problem-routines/lapack-least-squares-and-eigenvalue-problem-
   computational-routines/orthogonal-factorizations-lapack-computational-
   routines/geqrf.html

.. __: https://software.intel.com/content/www/us/en/develop/documentation/
   onemkl-developer-reference-c/top/lapack-routines/lapack-least-squares-and-
   eigenvalue-problem-routines/lapack-least-squares-and-eigenvalue-problem-
   computational-routines/orthogonal-factorizations-lapack-computational-
   routines/orgqr.html

.. __: https://www.gnu.org/software/gsl/


Compiling
---------

With ``make``
~~~~~~~~~~~~~

The easiest way to build the ``lapacke_demo`` executable is with GNU ``make``.
Make the ``demo_openblas`` target to dynamically link against OpenBLAS, the
``demo_lapacke`` target to dynamically link against LAPACKE, and the
``demo_mkl`` target to dynamically link against Intel MKL.

If any of the libraries are not located on the standard runtime library search
path, one can specify the base installation directory by overriding the values
of ``OPENBLAS_PATH``, ``LAPACKE_PATH``, and ``MKL_PATH``. For example, if one
built the `reference LAPACKE`__ from source, with the installation directory
``/path/to/LAPACKE``, the built libraries would be in ``/path/to/LAPACKE``
while the header files would be in ``/path/to/LAPACKE/LAPACKE/include``.
Linking against this copy of LAPACKE can be done with

.. code:: bash

   make LAPACKE_PATH=/path/to/LAPACKE demo_lapacke

Note that in the Makefile, several different directories are passed using
``-I``, ``-L``, and ``-Wl,-rpath`` to account for differences in directory
structure amongst common installation scenarios, without any assumptions on
the default include and runtime library search paths.

Optionally, one can also dynamically link against GSL, unless also linking to
Intel MKL [#]_, by also setting ``GSL_INCLUDE=1`` when calling ``make``. For
example, to dynamically link against OpenBLAS and GSL, where GSL is installed
at ``/path/to/GSL``, one can use

.. code:: bash

   make GSL_INCLUDE=1 GSL_PATH=/path/to/GSL demo_openblas

Of course, if the base GSL install directory is known to the system,
``GSL_PATH`` can be omitted.

.. __: https://github.com/Reference-LAPACK/lapack

.. [#] Dynamically linking against both Intel MKL and GSL is not possible, as
   the GSL and Intel MKL headers both declare the same identifiers, causing
   compilation to halt on account of identifier re-declaration being present. 

Manual compilation
~~~~~~~~~~~~~~~~~~

Assuming the compiler is ``gcc``, dynamic linking against OpenBLAS without
linking against GSL can be done with

.. code:: bash

   gcc -Wall -g -DLAPACKE_INCLUDE -I/path/to/OpenBLAS/include \
   -o lapacke_demo lapacke_demo.c -L/path/to/OpenBLAS/lib \
   -Wl,-rpath,/path/to/OpenBLAS/lib -lopenblas

Dynamically linking against LAPACKE without GSL can be done similarly with

.. code:: bash

   gcc -Wall -g -DLAPACKE_INCLUDE -I/path/to/LAPACKE/include \
   -o lapacke_demo lapacke_demo.c -L/path/to/LAPACKE/lib \
   -Wl,-rpath,/path/to/LAPACKE/lib -llapacke

Linking against Intel MKL requires a more complicated linker line. Assuming the
target machine has a 64-bit architecture, ``libpthread``, with 32-bit
``int`` [#]_
and no multithreading, the corresponding ``gcc`` invocation is

.. code:: bash

   gcc -Wall -g -DMKL_INCLUDE -I/path/to/MKL/include -m64 \
   -o lapacke_demo lapacke_demo.c -Wl,--no-as-needed \
   -L/path/to/MKL/lib -L/path/to/MKL/lib/intel64 \
   -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

There are two paths passed with ``-L`` as the former occurs if Intel MKL is
installed on Ubuntu using ``apt``. One can also use the
`Intel MKL Link Line Advisor`__ to generate the linker line given different
combinations of MKL version, OS, architecture, ``int`` size, threading layer,
etc. The linker line in the above invocation, excluding the additional
``-L/path/to/MKL/lib``, was generated using the Link Line Advisor.

.. __: https://software.intel.com/content/www/us/en/develop/tools/oneapi/
   components/onemkl/link-line-advisor.html

.. [#] Intel MKL also supports 64-bit ``int``, where one links against
   ``libmkl_intel_ilp64`` and specifies ``-DMKL_ILP64`` instead of linking
   against ``libmkl_intel_lp64``. The differences in ILP64 and LP64 are covered
   more in detail in the `dedicated article on them`__.

.. __: https://software.intel.com/content/www/us/en/develop/documentation/
   onemkl-linux-developer-guide/top/linking-your-application-with-the-intel-
   oneapi-math-kernel-library/linking-in-detail/linking-with-interface-
   libraries/using-the-ilp64-interface-vs-lp64-interface.html

Linker configuration
~~~~~~~~~~~~~~~~~~~~

``-Wl,-rpath,/path/to/my_lapacke_lib`` flags can be omitted only if you have
run ``ldconfig`` to update the ``ld.so`` cache or have included
``/path/to/OpenBLAS/lib`` in either ``/etc/ld.so.conf`` or a file in
``/etc/ld.so.conf.d``. Otherwise, omission results in a runtime linking error.

Execution
---------

Dead simple. After compilation, just run the demo with ``./lapacke_demo``. Some
printing to standard output will be done.