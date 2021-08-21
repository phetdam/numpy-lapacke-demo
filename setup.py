"""setup.py to build numpy-lapacke-demo C extension modules.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import os
import platform
from setuptools import Extension, find_packages, setup
import warnings

from npypacke import __package__, __version__

# package name and summary/short description
_PACKAGE_NAME = "numpy-lapacke-demo"
_PACKAGE_SUMMARY = """A small Python package demonstrating how to use LAPACKE \
and CBLAS with NumPy arrays in C extension modules."""
# general include dirs and required by all C extensions
_EXT_INCLUDE_DIRS = [f"{__package__}/include", np.get_include()]
# platform name + extra compile args for all C extensions, platform-dependent
_PLAT_NAME = platform.system()
if _PLAT_NAME == "Windows":
    _EXT_COMPILE_ARGS = ["/std:c11"]
else:
    _EXT_COMPILE_ARGS = ["-std=gnu11"]


def _get_ext_modules(env):
    """Returns a list of setuptools.Extension giving C extensions to build.

    Reads environment variables from mapping env, typically os.environ.
    UserWarnings raised and defaults are used if required variables are unset.
    Returned configuration may cause the setup function to fail on Windows if
    not linking against OpenBLAS. Also, PATH has path_extra appended on Windows
    and should be cleaned up after setup completes.

    Parameters
    ----------
    env: mapping
        Mapping representing the current environment. Typically os.environ.

    Returns
    -------
    ext_modules : list
        List of Extension instances to be sent to ext_modules kwargs of setup.
    path_extra : str or None
        On Windows, the OpenBLAS DLL is added to the Windows linker search path
        by modifying PATH. path_extra gives the exact string appended to PATH
        if not None, and can be cleaned from path by replacing path_extra in
        os.environ["PATH"] with the empty string.

    Raises
    ------
    RuntimeError
        Raised whenever environment configurations are incorrect/missing.
    """
    # get OpenBLAS, reference (Netlib) CBLAS/LAPACKE, MKL install paths. if
    # not specified in the environment, warn and use defaults.
    if "OPENBLAS_PATH" in env:
        OPENBLAS_PATH = env["OPENBLAS_PATH"]
    else:
        warnings.warn(
            "OPENBLAS_PATH not set. defaulting to OPENBLAS_PATH=/opt/OpenBLAS"
        )
        OPENBLAS_PATH = "/opt/OpenBLAS"
    if "NETLIB_PATH" in env:
        NETLIB_PATH = env["NETLIB_PATH"]
    else:
        warnings.warn("NETLIB_PATH not set. defaulting to NETLIB_PATH=/usr")
        NETLIB_PATH = "/usr"
    if "MKL_PATH" in env:
        MKL_PATH = env["MKL_PATH"]
    else:
        warnings.warn("MKL_PATH not set. defaulting to MKL_PATH=/usr")
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
    # if using MKL, check for unset MKL_INTERFACE_LAYER, MKL_THREADING_LAYER,
    # which control the mkl_rt interface. if unset, use default values
    if USE_MKL:
        if "MKL_INTERFACE_LAYER" not in env:
            warnings.warn(
                "MKL_INTERFACE_LAYER not set. setting "
                "MKL_INTERFACE_LAYER=GNU,ILP64"
            )
            env["MKL_INTERFACE_LAYER"] = "GNU,ILP64"
        if "MKL_THREADING_LAYER" not in env:
            warnings.warn(
                "MKL_THREADING_LAYER not set. setting "
                "MKL_THREADING_LAYER=SEQUENTIAL"
            )
            env["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    # if all are False, warn and default to OpenBLAS
    if not USE_OPENBLAS and not USE_NETLIB and not USE_MKL:
        warnings.warn(
            "neither of USE_OPENBLAS, USE_NETLIB, USE_MKL set to 1. "
            "defaulting to USE_OPENBLAS=1"
        )
        USE_OPENBLAS = True
    # if more than one is True, error (numpy function convenient here)
    if np.sum((USE_OPENBLAS, USE_NETLIB, USE_MKL)) > 1:
        raise RuntimeError(
            "only one of USE_OPENBLAS, USE_NETLIB, USE_MKL may be set"
        )
    # CBLAS + LAPACKE implementation include dirs (include_dirs), library dirs
    # (library_dirs), runtime library dirs (runtime_library_dirs), names of
    # libraries to link during extension building (libraries), preprocessor
    # macros to also define during compilation (define_macros), extra
    # compilation arguments that need to be passed (extra_compile_args)
    if USE_OPENBLAS:
        cblap_include_dirs = [f"{OPENBLAS_PATH}/include"]
        cblap_lib_dirs = [f"{OPENBLAS_PATH}/lib"]
        # on Windows, lib is not prepended to name of the LIB file
        cblap_lib_names = [
            "libopenblas" if _PLAT_NAME == "Windows" else "openblas"
        ]
        cblap_macros = [("OPENBLAS_INCLUDE", None)]
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
        cblap_macros = [("CBLAS_INCLUDE", None), ("LAPACKE_INCLUDE", None)]
        cblap_compile_args = []
    elif USE_MKL:
        cblap_include_dirs = [f"{MKL_PATH}/include", f"{MKL_PATH}/include/mkl"]
        cblap_lib_dirs = [
            f"{MKL_PATH}/lib/x86_64-linux-gnu", f"{MKL_PATH}/lib/intel64"
        ]
        cblap_lib_names = ["mkl_rt", "pthread", "m", "dl"]
        cblap_macros = [("MKL_INCLUDE", None)]
        cblap_compile_args = ["-m64"]
    # on Windows, MSVC doesn't support C99 _Complex type so we have to use the
    # corresponding MSVC complex types to define LAPACK complex types
    if _PLAT_NAME == "Windows":
        cblap_macros += [
            ("lapack_complex_float", "_Fcomplex"),
            ("lapack_complex_double", "_Dcomplex")
        ]
    # kwarg dict required by all C extensions calling CBLAS/LAPACKE routines
    cblap_build_kwargs = dict(
        include_dirs=cblap_include_dirs + _EXT_INCLUDE_DIRS,
        library_dirs=cblap_lib_dirs, runtime_library_dirs=cblap_lib_dirs,
        libraries=cblap_lib_names, define_macros=cblap_macros,
        extra_compile_args=cblap_compile_args + _EXT_COMPILE_ARGS
    )
    # on Windows, runtime_library_dirs corresponds to -Wl,-rpath for gcc and
    # MSVC has no corresponding option, so unset it. to search for DLL on
    # Windows we need to follow the standard search order, so we update PATH.
    if _PLAT_NAME == "Windows":
        del cblap_build_kwargs["runtime_library_dirs"]
        # PATH_EXTRA should be removed from PATH after _get_ext_modules returns
        path_extra = f";{OPENBLAS_PATH}/bin"
        env["PATH"] = env["PATH"] + path_extra
    # else we don't add path_extra to PATH variable, so set it to None
    else:
        path_extra = None
    # return C extension modules and path_extra. if path_extra is None, no
    # changes were made to PATH, else replace path_extra in PATH with "" later
    return [
        # npypacke.regression._linreg, providing LinearRegression class
        Extension(
            name="regression._linreg",
            sources=[f"{__package__}/regression/_linreg.c"],
            **cblap_build_kwargs
        ),
        # wrappers for unit testing internal C functions in _linreg.c
        Extension(
            name="regression._linreg_internal",
            sources=[f"{__package__}/regression/_linreg_internal.c"],
            include_dirs=_EXT_INCLUDE_DIRS,
            extra_compile_args=_EXT_COMPILE_ARGS
        ),
        # npypacke.solvers._mnewton, providing mnewton function
        Extension(
            name="solvers._mnewton",
            sources=[f"{__package__}/solvers/_mnewton.c"],
            **cblap_build_kwargs
        ),
        # wrappers for unit testing internal C functions in _mnewton.c
        Extension(
            name="solvers._mnewton_internal",
            sources=[f"{__package__}/solvers/_mnewton_internal.c"],
            include_dirs=_EXT_INCLUDE_DIRS,
            extra_compile_args=_EXT_COMPILE_ARGS
        )
    ], path_extra


def _setup():
    """Main setup method wrapping setuptools.setup."""
    # get long description from README.rst
    with open("README.rst") as rf:
        long_desc = rf.read().strip()
    # get Extension instances and path_extra
    ext_modules, path_extra = _get_ext_modules(os.environ)
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
        url="https://github.com/phetdam/numpy-lapacke-demo",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9"
        ],
        project_urls={
            "Source": "https://github.com/phetdam/numpy-lapacke-demo"
        },
        packages=find_packages(),
        python_requires=">=3.6",
        install_requires=["numpy>=1.19.1", "scipy>=1.5.2"],
        extras_require={"tests": ["pytest>=6.0.1", "scikit-learn>=0.23.2"]},
        ext_package=__package__,
        ext_modules=ext_modules
    )
    # if path_extra not None, it was appended to PATH, so remove it
    if path_extra:
        os.environ["PATH"] = os.environ["PATH"].replace(path_extra, "")


if __name__ == "__main__":
    _setup()