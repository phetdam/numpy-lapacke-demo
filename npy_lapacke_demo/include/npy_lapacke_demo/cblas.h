/**
 * @file cblas.h
 * @brief Header file to automatically handle Intel MKL or CBLAS includes.
 * 
 * `npy_lapacke_demo` can be linked with a reference CBLAS implementation,
 * OpenBLAS, or Intel MKL, but Intel MKL uses a different header file, `mkl.h`,
 * while OpenBLAS uses the CBLAS header `cblas.h`. This file includes the
 * correct header file given an appropriate preprocessor macro is defined.
 */

#ifndef NPY_LPK_CBLAS_H
#define NPY_LPK_CBLAS_H
// use CBLAS includes if passed this preprocessor flag (by -D)
#if defined(CBLAS_INCLUDE)
#include <cblas.h>
// use Intel MKL includes if passed this preprocessor flag
#elif defined(MKL_INCLUDE)
#include <mkl.h>
// else error, no LAPACKE includes specified
#else
// silence error squiggles in VS Code (__INTELLISENSE__ always defined)
#ifndef __INTELLISENSE__
#error "no CBLAS includes specified. try -DCBLAS_INCLUDE or -DMKL_INCLUDE"
#endif /* __INTELLISENSE__ */
#endif

// silence error squiggles in VS Code. use mkl.h since it also define MKL types
#ifdef __INTELLISENSE__
#include <mkl.h>
#endif /* __INTELLISENSE__ */

#endif /* NPY_LPK_CBLAS_H */