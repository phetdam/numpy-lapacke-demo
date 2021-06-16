# setup.py to build npy_lapacke_demo C extension modules.

import numpy as np
import os
from setuptools import Extension, find_packages, setup

from npy_lapacke_demo import __package__, __version__

# package name and summary/short description
_PACKAGE_NAME = __package__
_PACKAGE_SUMMARY = """A tiny Python package demonstrating LAPACKE calls on \
NumPy arrays in C extension modules.\
"""
# general include dirs and compile args required by all C extensions
_EXT_INCLUDE_DIRS = [f"{__package__}/include", np.get_include()]
_EXT_COMPILE_ARGS = ["-std=gnu11"]


def _get_ext_modules(env):
    """Returns a list of setuptools.Extension giving C extensions to build.

    Reads environment variables from mapping env, typically os.environ.

    Parameters
    ----------
    env: mapping
        Mapping representing the current environment. Typically os.environ.

    Returns
    -------
    list
        List of Extension instances to be sent to ext_modules kwargs of setup.

    Raises
    ------
    RuntimeError
        Raised whenever environment configurations are incorrect/missing.
    """
    # get OpenBLAS, LAPACKE, MKL install paths
    if "OPENBLAS_PATH" in env:
        OPENBLAS_PATH = env["OPENBLAS_PATH"]
    else:
        OPENBLAS_PATH = "/opt/OpenBLAS"
    if "LAPACKE_PATH" in env:
        LAPACKE_PATH = env["LAPACKE_PATH"]
    else:
        LAPACKE_PATH = "/usr"
    if "MKL_PATH" in env:
        MKL_PATH = env["MKL_PATH"]
    else:
        MKL_PATH = "/usr"
    # get flags to indicate which LAPACKE implementation to use
    if "USE_OPENBLAS" in env and env["USE_OPENBLAS"] == "1":
        USE_OPENBLAS = True
    else:
        USE_OPENBLAS = False
    if "USE_LAPACKE" in env and env["USE_LAPACKE"] == "1":
        USE_LAPACKE = True
    else:
        USE_LAPACKE = False
    if "USE_MKL" in env and env["USE_MKL"] == "1":
        USE_MKL = True
    else:
        USE_MKL = False
    # if all are False, error
    if not USE_OPENBLAS and not USE_LAPACKE and not USE_MKL:
        raise RuntimeError("none of USE_OPENBLAS, USE_LAPACKE, USE_MKL set")
    # if more than 1 is True, also error (numpy already imported)
    if np.sum((USE_OPENBLAS, USE_LAPACKE, USE_MKL)) > 1:
        raise RuntimeError(
            "only one of USE_OPENBLAS, USE_LAPACKE, USE_MKL may be set"
        )
    # LAPACKE implementation include dirs (include_dirs), library dirs
    # (library_dirs), runtime libary dirs (runtime_library_dirs), names of
    # libraries to link during extension building (libraries), preprocessor
    # macros to define during compilation (define_macros)
    if USE_OPENBLAS:
        lpke_include_dirs = [f"{OPENBLAS_PATH}/include"]
        lpke_lib_dirs = [f"{OPENBLAS_PATH}/lib"]
        lpke_lib_names = ["openblas"]
        lpke_macros = [("LAPACKE_INCLUDE", None)]
    elif USE_LAPACKE:
        lpke_include_dirs = [
            f"{LAPACKE_PATH}/include", f"{LAPACKE_PATH}/LAPACKE/include"
        ]
        lpke_lib_dirs = [
            LAPACKE_PATH, f"{LAPACKE_PATH}/lib",
            f"{LAPACKE_PATH}/lib/x86_64-linux-gnu"
        ]
        lpke_lib_names = ["lapacke"]
        lpke_macros = [("LAPACKE_INCLUDE", None)]
    elif USE_MKL:
        lpke_include_dirs = [f"{MKL_PATH}/include", f"{MKL_PATH}/include/mkl"]
        lpke_lib_dirs = [
            f"{MKL_PATH}/lib/x86_64-linux-gnu", f"{MKL_PATH}/lib/intel64"
        ]
        lpke_lib_names = [
            "mkl_intel_lp64", "mkl_sequential", "mkl_core", "pthread", "m", "dl"
        ]
        lpke_macros = [("MKL_INCLUDE", None)]
    # return C extension modules
    return [
        # npy_lapacke_demo.regression._linreg, providing LinearRegression class
        Extension(
            name="regression._linreg",
            sources=[f"{__package__}/regression/_linreg.c"],
            include_dirs=lpke_include_dirs + _EXT_INCLUDE_DIRS,
            library_dirs=lpke_lib_dirs, runtime_library_dirs=lpke_lib_dirs,
            libraries=lpke_lib_names, define_macros=lpke_macros,
            extra_compile_args=_EXT_COMPILE_ARGS
        )
    ]


def _setup():
    # get long description from README.rst
    with open("README.rst") as rf:
        long_desc = rf.read().strip()
    # run setuptools
    setup(
        name=_PACKAGE_NAME,
        version=__version__,
        description=_PACKAGE_SUMMARY,
        long_description=long_desc,
        long_description_content_type="text/x-rst",
        author="Derek Huang",
        author_email="djh458@stern.nyu.edu",
        license="MIT",
        url="https://github.com/phetdam/scipy_fastmin",
        packages=find_packages(),
        python_requires=">=3.6",
        install_requires=["numpy>=1.19.1", "scipy>=1.5.2"],
        ext_package=__package__,
        ext_modules=_get_ext_modules(os.environ)
    )


if __name__ == "__main__":
    _setup()