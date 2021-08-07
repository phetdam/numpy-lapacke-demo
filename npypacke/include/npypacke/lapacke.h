/**
 * @file lapacke.h
 * @brief Header file to automatically handle Intel MKL or LAPACKE includes.
 * 
 * `npypacke` can be linked with a reference LAPACKE implementation, OpenBLAS,
 * or Intel MKL, but Intel MKL uses a different header file, `mkl.h`, while
 * OpenBLAS uses the LAPACKE header `lapacke.h`. This file includes the correct
 * header file given an appropriate preprocessor macro is defined and defines
 * some types appropriately so that the user can simply write in terms of the
 * Intel MKL interface, whether or not Intel MKL is actually linked.
 */

#ifndef NPY_LPK_LAPACKE_H
#define NPY_LPK_LAPACKE_H
// if linking with Netlib LAPACKE
#if defined(LAPACKE_INCLUDE)
#include <lapacke.h>
// define MKL_INT used in extension modules as in mkl.h
#ifndef MKL_INT
#define MKL_INT int
#endif /* MKL_INT */
// else if linking with OpenBLAS CBLAS
#elif defined(OPENBLAS_INCLUDE)
#include <lapacke.h>
// OpenBLAS has blasint typedef, so define MKL_INT using blasint. for LAPACKE
// routines, should cast to lapack_int (same size as blasint in OpenBLAS)
#ifndef MKL_INT
#define MKL_INT blasint
#endif /* MKL_INT */
// else if linking with Intel MKL LAPACKE implementation
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

#endif /* NPY_LPK_LAPACKE_H */