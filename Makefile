# Makefile to build npy_lapacke_demo extensions, build + execute [gtest] tests.
# link against OpenBLAS by default but also allow LAPACKE and Intel MKL builds.

# package name
pkg_name               = npy_lapacke_demo
# directory for libgtest test runner code
gtest_dir              = gtest
# C extension module dependencies (nothing so far)
ext_deps               =
# Python source dependencies
py_deps                = $(wildcard $(pkg_name)/*.py)
# dependencies for test running code
gtest_deps             = $(wildcard $(gtest_dir)/*.cc)
# C and C++ compilers, of course
CC                     = gcc
CXX                    = g++
# set python; on docker specify PYTHON value externally using absolute path
PYTHON                 = python3
# general build flags to pass to setup.py build, build_ext
BUILD_FLAGS            =
# flags to pass to setup.py dist, bdist_wheel, sdist
DIST_FLAGS             =
# flags to indicate which LAPACKE implementation should be used
USE_OPENBLAS          ?=
USE_LAPACKE           ?=
USE_MKL               ?=
# location of OpenBLAS, LAPACKE, MKL install paths
OPENBLAS_PATH         ?= /opt/OpenBLAS
LAPACKE_PATH          ?= /usr
MKL_PATH              ?= /usr
# python compiler and linker flags for use when linking debug python into
# external C/C++ code; can be externally specified. gcc/g++ requires -fPIE.
PY_CFLAGS             ?= -fPIE $(shell python3d-config --cflags)
# ubuntu needs --embed, else -lpythonx.yd is omitted by --ldflags, which is a
# linker error. libpython3.8d is in /usr/lib/x86_64-linux-gnu for me.
PY_LDFLAGS            ?= $(shell python3d-config --embed --ldflags)
# google test installation path. libraries are in lib, includes in include.
GTEST_PATH             = /usr/local
# include and linker line for google test
gtest_include_line     = -I$(GTEST_PATH)/include
gtest_link_line        = -L$(GTEST_PATH)/lib \
	-L$(GTEST_PATH)/lib/x86_64-linux-gnu -Wl,-rpath,$(GTEST_PATH)/lib \
	-Wl,-rpath,$(GTEST_PATH)lib/x86_64-linux-gnu -lgtest -lgtest_main
# g++ compile flags for gtest runner
GTEST_CFLAGS   = -I$(GTEST_PATH)/include $(PY_CFLAGS)
# g++ linker flags for compiling gtest runner
GTEST_LDFLAGS  = -L$(GTEST_PATH)/lib -Wl,-rpath,$(GTEST_PATH)/lib \
	-lgtest -lgtest_main $(PY_LDFLAGS)
# flags to pass to the gtest test runner
RUNNER_FLAGS   =

# phony targets
.PHONY: check clean dummy

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
inplace: $(ext_deps)
	$(PYTHON) setup.py build_ext --inplace $(BUILD_FLAGS)

# build test runner and run gtest unit tests. show flags passed to g++
check: $(gtest_deps) inplace
	$(CXX) $(GTEST_CFLAGS) -o runner $(gtest_deps) $(GTEST_LDFLAGS)
	@./runner $(RUNNER_FLAGS)

# make source and wheel, linking to OpenBLAS
dist: build
	USE_OPENBLAS=1 $(PYTHON) setup.py sdist bdist_wheel $(DIST_FLAGS)

# make just wheel, linking to OpenBLAS
bdist_wheel: build
	USE_OPENBLAS=1 $(PYTHON) setup.py bdist_wheel $(DIST_FLAGS)

# make just sdist
sdist: $(ext_deps) $(py_deps)
	$(PYTHON) setup.py sdist $(DIST_FLAGS)