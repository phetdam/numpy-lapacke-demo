/**
 * @file cblas.h
 * @brief Header file to handle Intel MKL, OpenBLAS, or Netlib includes.
 * 
 * `npypacke` can be linked with a reference CBLAS implementation, OpenBLAS, or
 * Intel MKL, but Intel MKL uses a different header file, `mkl.h`, while
 * OpenBLAS uses the CBLAS header `cblas.h`. This file includes the correct
 * header file given an appropriate preprocessor macro is defined, and defines
 * some types appropriately so that the user can simply write in terms of the
 * Intel MKL interface, whether or not Intel MKL is actually linked.
 */

#ifndef NPY_LPK_CBLAS_H
#define NPY_LPK_CBLAS_H
// if linking with Netlib CBLAS
#if defined(CBLAS_INCLUDE)
#include <cblas.h>
// define MKL_INT used in extension modules as in mkl.h
#ifndef MKL_INT
#define MKL_INT int
#endif /* MKL_INT */
// else if linking with OpenBLAS CBLAS
#elif defined(OPENBLAS_INCLUDE)
#include <cblas.h>
// OpenBLAS has blasint typedef, so define MKL_INT using blasint
#ifndef MKL_INT
#define MKL_INT blasint
#endif /* MKL_INT */
// else if linking to Intel MKL
#elif defined(MKL_INCLUDE)
#include <mkl.h>
// else error, no LAPACKE includes specified
#else
// silence error squiggles in VS Code (__INTELLISENSE__ always defined)
#ifndef __INTELLISENSE__
#error "no CBLAS includes specified. try -D(CBLAS|OPENBLAS|MKL)_INCLUDE"
#endif /* __INTELLISENSE__ */
#endif

// silence error squiggles in VS Code. use mkl.h since it also define MKL types
#ifdef __INTELLISENSE__
#include <mkl.h>
#endif /* __INTELLISENSE__ */

#endif /* NPY_LPK_CBLAS_H */