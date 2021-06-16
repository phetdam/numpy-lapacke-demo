/**
 * @file lapacke.h
 * @brief Header file to automatically handle Intel MKL or LAPACKE includes.
 * 
 * `npy_lapacke_demo` can be linked with either a LAPACKE implementation,
 * OpenBLAS, or Intel MKL, but Intel MKL uses a different header file, `mkl.h`,
 * while OpenBLAS uses the LAPACKE header `lapacke.h`. This file includes the
 * correct header file given an appropriate preprocessor macro is defined.
 */

#ifndef NPY_LPK_LAPACKE_H
#define NPY_LPK_LAPACKE_H
// use LAPACKE includes if passed this preprocessor flag (by -D)
#if defined(LAPACKE_INCLUDE)
#include <lapacke.h>
// use Intel MKL includes if passed this preprocessor flag
#elif defined(MKL_INCLUDE)
#include <mkl.h>
// else error, no LAPACKE includes specified
#else
// silence error squiggles in VS Code (__INTELLISENSE__ always defined)
#ifndef __INTELLISENSE__
#error "no LAPACKE includes specified. try -DLAPACKE_INCLUDE or -DMKL_INCLUDE"
#endif /* __INTELLISENSE__ */
#endif

// silence error squiggles in VS Code. use mkl.h since it also define MKL types
#ifdef __INTELLISENSE__
#include <mkl.h>
#endif /* __INTELLISENSE__ */

#endif /* NPY_LAPACKE_H */