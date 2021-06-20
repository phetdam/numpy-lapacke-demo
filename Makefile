# Makefile to build npy_lapacke_demo extensions, build + execute [gtest] tests.
# link against OpenBLAS by default but also allow Netlib and Intel MKL builds.
#
# to specify a particular CBLAS + LAPACKE implementation to link against, set
# one of USE_OPENBLAS, USE_NETLIB, USE_MKL to 1 when running make. if none are
# specified, setup.py will warn and act as if USE_OPENBLAS=1 was specified.
#
# see setup.py _get_ext_modules for details on defaults.

# package name
pkg_name = npy_lapacke_demo
# directory for libgtest test runner code
gtest_dir = gtest
# C extension module dependencies
ext_deps = $(wildcard $(pkg_name)/regression/*.c)
# Python source dependencies. important to include setup.py, which does config
py_deps = setup.py $(wildcard $(pkg_name)/*.py) \
	$(wildcard $(pkg_name)/regression/*.py)
# C and C++ compilers, of course
CC = gcc
CXX = g++
# set python; on docker specify PYTHON value externally using absolute path
PYTHON = python3
# since setup.py won't rebuild if re-run without changing the source, i.e. if
# you just pass USE_OPENBLAS=1 instead of USE_NETLIB=1, set to 1 to rebuild.
FORCE_BUILD ?=
# general build flags to pass to setup.py build, build_ext
BUILD_FLAGS =
# flags to pass to setup.py dist, bdist_wheel, sdist
DIST_FLAGS =
# default MKL interface layer to use with single dynamic library. other options
# include "LP64", "ILP64", or "GNU,LP64". note that without the GNU prefix
# calls to Intel MKL functions result is ugly crashes! also, since we want the
# variable to be seen in the environment, we need to export it.
export MKL_INTERFACE_LAYER ?= GNU,ILP64
# default MKL threading layer to use with single dynamic library. other options
# include "INTEL" for Intel threading, "GNU" for libgomp threading, "PGI" for
# PGI threading (idk what this is), "TBB" for Intel TBB threading. exported.
export MKL_THREADING_LAYER ?= SEQUENTIAL
# OpenBLAS, (reference) CBLAS + LAPACKE, MKL install paths. exported to env.
export OPENBLAS_PATH ?= /opt/OpenBLAS/
export NETLIB_PATH ?= /usr
export MKL_PATH ?= /usr
# whether or not to expose the EXPOSED_* methods in the C extensions. useful
# for unit testing methods typically private to the modules. on by default.
export EXPOSE_INTERNAL ?= 1
# arguments to pass to pytest. default here shows skipped, xfailed, xpassed,
# and passed tests that print output in the brief summary report.
PYTEST_ARGS ?= -rsxXP

# to force setup.py to rebuild, add clean as a target, which is a phony target
ifeq ($(REBUILD), 1)
py_deps += clean
endif

# phony targets. note sdist just copies files.
.PHONY: check clean dummy sdist

# triggered if no target is provided
dummy:
	@echo "Please specify a target to build."

# removes local build, dist, egg-info
clean:
	@rm -vrf build
	@rm -vrf $(pkg_name).egg-info
	@rm -vrf dist

# build extension module locally in ./build from source files with setup.py
build: $(ext_deps) $(py_deps)
	$(PYTHON) setup.py build $(BUILD_FLAGS)

# build extension modules in-place with build_ext --inplace. in-place means
# the shared objects will be in the same directory as the respective sources.
inplace: build
	$(PYTHON) setup.py build_ext --inplace $(BUILD_FLAGS)

# just run pytest with arguments given by PYTEST_ARGS
check: inplace
	pytest $(PYTEST_ARGS)

# make source and wheel, no access to EXPOSED_* functions.
dist: build
	EXPOSE_INTERNAL= $(PYTHON) setup.py sdist bdist_wheel $(DIST_FLAGS)

# make wheel, no access to EXPOSED_* functions.
bdist_wheel: $(ext_deps) $(py_deps)
	EXPOSE_INTERNAL= $(PYTHON) setup.py bdist_wheel $(DIST_FLAGS)

# make just sdist. ok to not set EXPOSED_*; only copies files. set
# USE_OPENBLAS=1 to suppress warnings (doesn't have any effect).
sdist:
	USE_OPENBLAS=1 $(PYTHON) setup.py sdist $(DIST_FLAGS)