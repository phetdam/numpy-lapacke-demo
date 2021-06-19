# Makefile to build npy_lapacke_demo extensions, build + execute [gtest] tests.
# link against OpenBLAS by default but also allow Netlib and Intel MKL builds.

# package name
pkg_name               = npy_lapacke_demo
# directory for libgtest test runner code
gtest_dir              = gtest
# C extension module dependencies
ext_deps               = $(wildcard $(pkg_name)/regression/*.c)
# Python source dependencies. important to include setup.py, which does config
py_deps                = setup.py $(wildcard $(pkg_name)/*.py) \
	$(wildcard $(pkg_name)/regression/*.py)
# dependencies for test running code. since we would also like to directly test
# some static functions within the C extension modules since they aren't
# accessible from Python (not in module's method table, not safely castable to
# PyCFunction), we directly include them in the test runner. the runner is
# built with EXPOSE_INTERNAL so the relevant functions are not static.
gtest_deps             = $(wildcard $(gtest_dir)/*.cc) $(ext_deps)
# C and C++ compilers, of course
CC                     = gcc
CXX                    = g++
# set python; on docker specify PYTHON value externally using absolute path
PYTHON                 = python3
# general build flags to pass to setup.py build, build_ext
BUILD_FLAGS            =
# flags to pass to setup.py dist, bdist_wheel, sdist
DIST_FLAGS             =
# flags to indicate which CBLAS + LAPACKE implementation should be used
USE_OPENBLAS          ?=
USE_NETLIB            ?=
USE_MKL               ?=
# default MKL interface layer to use with single dynamic library. other options
# include "LP64", "ILP64", or "GNU,LP64". note that without the GNU prefix
# calls to Intel MKL functions result is ugly crashes!
MKL_INTERFACE_LAYER   ?= GNU,ILP64
# default MKL threading layer to use with single dynamic library. other options
# include "INTEL" for Intel threading, "GNU" for libgomp threading, "PGI" for
# PGI threading (not sure what this is), "TBB" for Intel TBB threading.
MKL_THREADING_LAYER   ?= SEQUENTIAL
# location of OpenBLAS, (reference) CBLAS + LAPACKE, MKL install paths
OPENBLAS_PATH         ?= /opt/OpenBLAS
NETLIB_PATH           ?= /usr
MKL_PATH              ?= /usr
# whether or not to expose the EXPOSED_* methods in the C extensions. useful
# for unit testing methods typically private to the modules.
EXPOSE_INTERNAL       ?=
# arguments to pass to pytest. default here shows skipped, xfailed, xpassed,
# and passed tests that print output in the brief summary report.
PYTEST_ARGS           ?= -rsxXP

# phony targets. always rebuild since we might want to pass different flags to
# make build but without changing any of the source code.
.PHONY: build check clean dummy

# triggered if no target is provided
dummy:
	@echo "Please specify a target to build."

# removes local build, dist, egg-info
clean:
	@rm -vrf build
	@rm -vrf $(pkg_name).egg-info
	@rm -vrf dist

# build extension module locally in ./build from source files with setup.py
# triggers when any of the files that are required are touched/modified.
build: $(ext_deps) $(py_deps)
	$(PYTHON) setup.py build $(BUILD_FLAGS)

# build extension modules in-place with build_ext --inplace. in-place means
# the shared objects will be in the same directory as the respective sources.
inplace: build
	$(PYTHON) setup.py build_ext --inplace $(BUILD_FLAGS)

# just run pytest with arguments given by PYTEST_ARGS
check: inplace
	pytest $(PYTEST_ARGS)

# make source and wheel, linking to OpenBLAS. note we explicitly depend on
# ext_deps, py_deps so we don't have to manually specify USE_OPENBLAS=1
dist: $(ext_deps) $(py_deps)
	USE_OPENBLAS=1 $(PYTHON) setup.py sdist bdist_wheel $(DIST_FLAGS)

# make just wheel, linking to OpenBLAS. use of explicit deps same as dist.
bdist_wheel: $(ext_deps) $(py_deps)
	USE_OPENBLAS=1 $(PYTHON) setup.py bdist_wheel $(DIST_FLAGS)

# make just sdist
sdist: $(ext_deps) $(py_deps)
	$(PYTHON) setup.py sdist $(DIST_FLAGS)