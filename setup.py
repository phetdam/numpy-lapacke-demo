# setup.py to build npy_lapacke_demo C extension modules.

import numpy as np
import os
from setuptools import Extension, find_packages, setup

from npy_lapacke_demo import __package__, __version__

# package name and summary/short description
_PACKAGE_NAME = __package__
_PACKAGE_SUMMARY = """A tiny Python package demonstrating how to use LAPACKE \
and CBLAS with NumPy arrays in C extension modules.\
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
    # get OpenBLAS, reference (Netlib) CBLAS/LAPACKE, MKL install paths
    if "OPENBLAS_PATH" in env:
        OPENBLAS_PATH = env["OPENBLAS_PATH"]
    else:
        OPENBLAS_PATH = "/opt/OpenBLAS"
    if "NETLIB_PATH" in env:
        NETLIB_PATH = env["NETLIB_PATH"]
    else:
        NETLIB_PATH = "/usr"
    if "MKL_PATH" in env:
        MKL_PATH = env["MKL_PATH"]
    else:
        MKL_PATH = "/usr"
    # get flags to indicate which CBLAS + LAPACKE implementations to use
    if "USE_OPENBLAS" in env and env["USE_OPENBLAS"] == "1":
        USE_OPENBLAS = True
    else:
        USE_OPENBLAS = False
    if "USE_NETLIB" in env and env["USE_NETLIB"] == "1":
        USE_NETLIB = True
    else:
        USE_NETLIB = False
    if "USE_MKL" in env and env["USE_MKL"] == "1":
        USE_MKL = True
    else:
        USE_MKL = False
    # if all are False, error
    if not USE_OPENBLAS and not USE_NETLIB and not USE_MKL:
        raise RuntimeError("none of USE_OPENBLAS, USE_NETLIB, USE_MKL set")
    # if more than 1 is True, also error (numpy already imported)
    if np.sum((USE_OPENBLAS, USE_NETLIB, USE_MKL)) > 1:
        raise RuntimeError(
            "only one of USE_OPENBLAS, USE_NETLIB, USE_MKL may be set"
        )
    # get flag indicating whether to build with access to internal extension
    # module functions. EXPOSE_INTERNAL results in some functions that are
    # static being declared as non-static, which makes them accessible from a
    # test runner program that also links against libpython3.x.
    if ("EXPOSE_INTERNAL" in env and env["EXPOSE_INTERNAL"] == "1"):
        cblap_macros = [("EXPOSE_INTERNAL", None)]
    else:
        cblap_macros = []
    # CBLAS + LAPACKE implementation include dirs (include_dirs), library dirs
    # (library_dirs), runtime libary dirs (runtime_library_dirs), names of
    # libraries to link during extension building (libraries), preprocessor
    # macros to also define during compilation (define_macros), extra
    # compilation arguments that need to be passed (extra_compile_args)
    if USE_OPENBLAS:
        cblap_include_dirs = [f"{OPENBLAS_PATH}/include"]
        cblap_lib_dirs = [f"{OPENBLAS_PATH}/lib"]
        cblap_lib_names = ["openblas"]
        cblap_macros += [("OPENBLAS_INCLUDE", None)]
        cblap_compile_args = []
    elif USE_NETLIB:
        cblap_include_dirs = [
            f"{NETLIB_PATH}/include", f"{NETLIB_PATH}/include/x86_64-linux-gnu"
            f"{NETLIB_PATH}/CBLAS/include", f"{NETLIB_PATH}/LAPACKE/include",
        ]
        cblap_lib_dirs = [
            NETLIB_PATH, f"{NETLIB_PATH}/lib",
            f"{NETLIB_PATH}/lib/x86_64-linux-gnu"
        ]
        cblap_lib_names = ["blas", "lapacke"]
        cblap_macros += [("CBLAS_INCLUDE", None), ("LAPACKE_INCLUDE", None)]
        cblap_compile_args = []
    elif USE_MKL:
        cblap_include_dirs = [f"{MKL_PATH}/include", f"{MKL_PATH}/include/mkl"]
        cblap_lib_dirs = [
            f"{MKL_PATH}/lib/x86_64-linux-gnu", f"{MKL_PATH}/lib/intel64"
        ]
        cblap_lib_names = ["mkl_rt", "pthread", "m", "dl"]
        cblap_macros += [("MKL_INCLUDE", None)]
        cblap_compile_args = ["-m64"]
    # return C extension modules
    return [
        # npy_lapacke_demo.regression._linreg, providing LinearRegression class
        Extension(
            name="regression._linreg",
            sources=[f"{__package__}/regression/_linreg.c"],
            include_dirs=cblap_include_dirs + _EXT_INCLUDE_DIRS,
            library_dirs=cblap_lib_dirs, runtime_library_dirs=cblap_lib_dirs,
            libraries=cblap_lib_names, define_macros=cblap_macros,
            extra_compile_args=cblap_compile_args + _EXT_COMPILE_ARGS
        )
    ]


def _setup():
    """Main setup method wrapping setuptools.setup."""
    # get long description from README.rst
    with open("README.rst") as rf:
        long_desc = rf.read().strip()
    # run setuptools setup
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
        install_requires=["numpy>=1.19.1"],
        extras_require={"tests": ["scikit-learn>=0.23.2"]},
        ext_package=__package__,
        ext_modules=_get_ext_modules(os.environ)
    )


if __name__ == "__main__":
    _setup()