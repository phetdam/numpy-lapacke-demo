"""setup.py to build numpy-lapacke-demo C extension modules.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import glob
import os
import platform
import shutil
import warnings

import numpy as np
from setuptools import Extension, find_packages, setup

# pylint: disable=redefined-builtin
from npypacke import __package__, __version__
# pylint: enable=redefined-builtin

# package name and summary/short description
_PACKAGE_NAME = "numpy-lapacke-demo"
_PACKAGE_SUMMARY = """A small Python package demonstrating how to use LAPACKE \
and CBLAS with NumPy arrays in C extension modules."""
# general include dirs and required by all C extensions
_EXT_INCLUDE_DIRS = [f"{__package__}/include", np.get_include()]
# platform name + extra compile args for all C extensions, platform-dependent
_PLAT_NAME = platform.system()
if _PLAT_NAME == "Windows":
    _EXT_COMPILE_ARGS = ["/std:c11", "/d2FH4-"]
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
    delocated : str
        Directory where external shared libraries are located that are to be
        copied (delocated) into wheel for distribution, as we cannot link
        assume the existence of shared libs when distributing. None if the
        DELOCATED environment vairable is not set.

    Raises
    ------
    RuntimeError
        Raised whenever environment configurations are incorrect/missing.
    """
    # get OpenBLAS, reference (Netlib) CBLAS/LAPACKE, MKL install paths. if
    # not specified in the environment, warn and use defaults.
    if "OPENBLAS_PATH" in env:
        openblas_path = env["OPENBLAS_PATH"]
    else:
        warnings.warn(
            "OPENBLAS_PATH not set. defaulting to OPENBLAS_PATH=/opt/OpenBLAS"
        )
        openblas_path = "/opt/OpenBLAS"
    if "NETLIB_PATH" in env:
        netlib_path = env["NETLIB_PATH"]
    else:
        warnings.warn("NETLIB_PATH not set. defaulting to NETLIB_PATH=/usr")
        netlib_path = "/usr"
    if "MKL_PATH" in env:
        mkl_path = env["MKL_PATH"]
    else:
        warnings.warn("MKL_PATH not set. defaulting to MKL_PATH=/usr")
        mkl_path = "/usr"
    # get flags to indicate which CBLAS + LAPACKE implementations to use
    if "USE_OPENBLAS" in env and env["USE_OPENBLAS"] == "1":
        use_openblas = True
    else:
        use_openblas = False
    if "USE_NETLIB" in env and env["USE_NETLIB"] == "1":
        use_netlib = True
    else:
        use_netlib = False
    if "USE_MKL" in env and env["USE_MKL"] == "1":
        use_mkl = True
    else:
        use_mkl = False
    # support copying shared libs into package. most relevant for Windows.
    # indicate wheel repair tool to use instead of DELOCATED if possible.
    if "DELOCATED" in env:
        warnings.warn(
            "DELOCATED set. specified shared libs will be copied into "
            f"{__package__}. if you are building only for your machine, do "
            "NOT set DELOCATED and link to your machine's shared libs."
        )
        if _PLAT_NAME == "Linux":
            wheel_repair_tool = "auditwheel"
        elif _PLAT_NAME == "Darwin":
            wheel_repair_tool = "delocate"
        else:
            wheel_repair_tool = None
        if _PLAT_NAME != "Windows" and wheel_repair_tool is not None:
            warnings.warn(
                "non-Windows platform detected. if you wish to build a "
                f"redistributable wheel, please use {wheel_repair_tool}"
            )
        delocated = env["DELOCATED"].split(os.pathsep)
    else:
        delocated = None
    # if using MKL, check for unset MKL_INTERFACE_LAYER, MKL_THREADING_LAYER,
    # which control the mkl_rt interface. if unset, use default values
    if use_mkl:
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
    if not use_openblas and not use_netlib and not use_mkl:
        warnings.warn(
            "neither of USE_OPENBLAS, USE_NETLIB, USE_MKL set to 1. "
            "defaulting to USE_OPENBLAS=1"
        )
        use_openblas = True
    # if more than one is True, error (numpy function convenient here)
    if np.sum((use_openblas, use_netlib, use_mkl)) > 1:
        raise RuntimeError(
            "only one of USE_OPENBLAS, USE_NETLIB, USE_MKL may be set"
        )
    # CBLAS + LAPACKE implementation include dirs (include_dirs), library dirs
    # (library_dirs), runtime library dirs (runtime_library_dirs), names of
    # libraries to link during extension building (libraries), preprocessor
    # macros to also define during compilation (define_macros), extra
    # compilation arguments that need to be passed (extra_compile_args)
    if use_openblas:
        cblap_include_dirs = [f"{openblas_path}/include"]
        cblap_lib_dirs = [f"{openblas_path}/lib"]
        # on Windows, lib is not prepended to name of the LIB file
        cblap_lib_names = [
            "libopenblas" if _PLAT_NAME == "Windows" else "openblas"
        ]
        cblap_macros = [("OPENBLAS_INCLUDE", None)]
        cblap_compile_args = []
    elif use_netlib:
        cblap_include_dirs = [
            f"{netlib_path}/include", f"{netlib_path}/include/x86_64-linux-gnu"
            f"{netlib_path}/CBLAS/include", f"{netlib_path}/LAPACKE/include",
        ]
        cblap_lib_dirs = [
            netlib_path, f"{netlib_path}/lib",
            f"{netlib_path}/lib/x86_64-linux-gnu"
        ]
        cblap_lib_names = ["blas", "lapacke"]
        cblap_macros = [("CBLAS_INCLUDE", None), ("LAPACKE_INCLUDE", None)]
        cblap_compile_args = []
    elif use_mkl:
        cblap_include_dirs = [f"{mkl_path}/include", f"{mkl_path}/include/mkl"]
        cblap_lib_dirs = [
            f"{mkl_path}/lib/x86_64-linux-gnu", f"{mkl_path}/lib/intel64"
        ]
        cblap_lib_names = ["mkl_rt", "pthread", "m", "dl"]
        cblap_macros = [("MKL_INCLUDE", None)]
        cblap_compile_args = ["-m64"]
    # if DELOCATED not None and not on Windows, cblap_lib_dirs set to base
    # package path. this is because Windows often has .lib import libraries
    # which we still need to refer to in order to link against DLLs; we handle
    # the DLL search path later after initializing cblap_build_kwargs
    if delocated is not None and _PLAT_NAME != "Windows":
        cblap_lib_dirs = [f"{__package__}"]
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
        # else PATH will grow. if DELOCATED not None, very simple case.
        if delocated is not None:
            path_extra = f";{__package__}"
        elif use_openblas:
            path_extra = f";{openblas_path}/bin"
        # note: USE_NETLIB and USE_MKL cases have not been tested!
        else:
            path_extra = ";".join([""] + cblap_lib_dirs)
        env["PATH"] = env["PATH"] + path_extra
    # else we don't add path_extra to PATH variable, so set it to None
    else:
        path_extra = None
    # return C extension modules, path_extra, files to pass to data_files. if
    # path_extra is None, no changes were made to PATH, else replace path_extra
    # in PATH with "" after _get_ext_modules returns.
    ext_modules = [
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
    ]
    return ext_modules, path_extra, delocated


def _setup():
    """Main setup method wrapping setuptools.setup."""
    # get long description from README.rst
    with open("README.rst") as rf:
        long_desc = rf.read().strip()
    # get Extension instances, path_extra, and shared objects to copy
    ext_modules, path_extra, deloc_globs = _get_ext_modules(os.environ)
    # if deloc_globs given, copy into top-level package, matching globs. note
    # that file metadata may NOT be preserved in all cases!
    if deloc_globs is not None:
        for deloc_glob in deloc_globs:
            deloc_files = glob.glob(deloc_glob)
            # raise exception if glob pattern match fails
            if len(deloc_files) == 0:
                raise FileNotFoundError(
                    f"can't match glob pattern {deloc_glob} in DELOCATED"
                )
            # else just copy
            for deloc_file in deloc_files:
                shutil.copy2(deloc_file, f"{__package__}")
    # if deloc_files is not None, copy all into top-level package dir
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
        ext_modules=ext_modules,
        package_data={f"{__package__}": ["*.so", "*.dll", "*.dylib"]}
    )
    # if path_extra not None, it was appended to PATH, so remove it
    if path_extra:
        os.environ["PATH"] = os.environ["PATH"].replace(path_extra, "")


if __name__ == "__main__":
    _setup()
