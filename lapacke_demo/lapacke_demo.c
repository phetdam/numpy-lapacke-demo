/**
 * @file lapacke_demo.c
 * @brief Demonstrate LAPACKE functionality by performing QR decomposition.
 * 
 * Linking Intel MKL can be a pain, but the Intel Math Kernel Library Link Line
 * Advisor helps generate the linker line for you, which on Ubuntu 20.04 LTS is
 * 
 *     -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread
 *     -lmkl_core -lgomp -lpthread -lm -ldl
 * 
 * Note we replace -lmkl_intel_lp64 with -lmkl_intel_ilp64 for 64-bit ints.
 * Read more in Intel's Using the ILP64 Interface vs. LP64 Interface article.
 * The compiler flag required on Ubuntu 20.04 LTS is -m64. OOTB, Intel MKL does
 * not work with GSL since CBLAS declarations are re-declared, so all
 * GSL-related code is included only if we are not linking with Intel MKL.
 * 
 * Both OpenBLAS and LAPACKE can be linked, although without ldconfig,
 * `-L/path/to/OpenBLAS -Wl,-rpath -Wl,/path/to/OpenBLAS` must be provided in
 * the former case since OpenBLAS by default is not installed in a directory
 * that is scanned during linking and during runtime.
 * 
 * To link to CBLAS and LAPACKE separately, one should link against liblapacke
 * and libblas, using a proper runtime linker path specification.
 */

#include <stdlib.h>
#include <stdio.h>

// use LAPACKE includes if passed this preprocessor flag (by -D)
#if defined(LAPACKE_INCLUDE)
#include <lapacke.h>
/**
 * replace use of MKL_INT with normal int. int size changes if we link against
 * libmkl_intel_ilp64 (64 bit int) instead of libmkl_intel_lp64 (32 bit int),
 * so it's better to just use MKL_INT and #define it as int otherwise.
 */
#define MKL_INT int
// use Intel MKL includes if passed this preprocessor flag
#elif defined(MKL_INCLUDE)
#include <mkl.h>
// else error, no BLAS + LAPACK includes specified
#else
// silence error squiggles in VS Code (__INTELLISENSE__ always defined)
#ifndef __INTELLISENSE__
#error "no LAPACKE includes specified. try -DLAPACKE_INCLUDE or -DMKL_INCLUDE"
#endif /* __INTELLISENSE__ */
#endif

// optionally include GSL headers
#ifdef GSL_INCLUDE
// can't link GSL with Intel MKL since identifiers are re-declared in headers
#ifdef MKL_INCLUDE
#error "-DMKL_INCLUDE cannot be specified with -DGSL_INCLUDE"
#else
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#endif /* MKL_INCLUDE */
#endif /* GSL_INCLUDE */

// silence error squiggles in VS Code
#ifdef __INTELLISENSE__
#include <mkl.h>
#endif /* __INTELLISENSE__ */

// defines for the main (variable-sized arrays can't use initializers)
#define NROW 4
#define NCOL 3
#define RANK 3
#define SIZE (NROW * NCOL)
// the matrix initializer we will use
#define MAT_INIT {5, 9, 1, -4, -1, 3, 4, -2, -9, -3, 7, 6}

/**
 * Simple function to print out a matrix laid out in row-major as a vector.
 * 
 * @param mat `double *` giving elements in the matrix, length `nrow` * `ncol`.
 * @param nrow `int` giving number of rows
 * @param ncol `int` giving number of columns
 * @returns -1 on error, 0 on success
 */
static int print_matrix(const double *mat, int nrow, int ncol)
{
  if (mat == NULL || nrow < 1 || ncol < 1) {
    return -1;
  }
  for (int i = 0; i < nrow; i++) {
    printf("[ ");
    for (int j = 0; j < ncol; j++) {
      printf("% .3e ", mat[i * ncol + j]);
    }
    printf("]\n");
  }
  return 0;
}

int main(int argc, char **argv)
{
  /**
   *                         [  5  9  1 ]
   * represents the matrix   [ -4 -1  3 ]
   *                         [  4 -2 -9 ]
   *                         [ -3  7  6 ]
   * 
   * obviously larger arrays should be put on the heap. it's even better to
   * have aligned memory with mkl_malloc or posix_memalign.
   */
  double mat[SIZE] = MAT_INIT;
  // the (NCOL, NCOL) upper triangular matrix R for the QR of mat.
  double u_mat[NCOL * NCOL];
  // the resulting (NROW, NCOL) orthogonal matrix Q for mat/temp matrix.
  double o_mat[SIZE] = MAT_INIT;
  // coefficients for the elementary reflectors
  double tau[RANK];
  // zero elements below main diagonal of u_mat with zeros + print mat
  for (MKL_INT i = 1; i < NCOL; i++) {
    for (MKL_INT j = 0; j < i; j++) {
      u_mat[i * NCOL + j] = 0;
    }
  }
  printf("original matrix A:\n");
  print_matrix(mat, NROW, NCOL);
  /**
   * perform QR on mat (on o_mat really, which is overwritten). now, o_mat's
   * upper triangle is R from the QR factorization of mat, where the remaining
   * m - i elements per ith column (1-indexing) give the nonzero (excluding
   * leading 1) trailing parts of the orthogonal vectors used to define the
   * RANK Householder matrices whose product is Q from the QR factorization.
   */
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, NROW, NCOL, o_mat, NCOL, tau);
  // copy appropriate elements into u_mat + print
  for (MKL_INT i = 0; i < NCOL; i++) {
    for (MKL_INT j = i; j < NCOL; j++) {
      u_mat[i * NCOL + j] = o_mat[i * NCOL + j];
    }
  }
  printf("\nR from QR of A:\n");
  print_matrix(u_mat, NCOL, NCOL);
  // retrieve Q from QR of mat from o_mat (holds Householder vectors) + print
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, NROW, NCOL, RANK, o_mat, NCOL, tau);
  printf("\nQ from QR of A:\n");
  print_matrix(o_mat, NROW, NCOL);

// silence error squiggles in VS Code
#ifndef __INTELLISENSE__
// include GSL code generating a random Gaussian matrix ifdef GSL_INCLUDE
#ifdef GSL_INCLUDE
  printf("\nGSL enabled: -DGSL_INCLUDE specified\n\n");
  // allocate Mersenne Twister PRNG + seed it
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
  if (rng == NULL) {
    fprintf(stderr, "FATAL: unable to allocate new gsl_rng\n");
    return EXIT_FAILURE;
  }
  gsl_rng_set(rng, 7L);
  // initialize random matrix with standard Gaussian entries + print
  double r_mat[NROW * NCOL];
  for (int i = 0; i < NROW; i++) {
    for (int j = 0; j < NCOL; j++) {
      r_mat[i * NCOL + j] = gsl_ran_gaussian(rng, 1);
    }
  }
  printf("random Gaussian matrix:\n");
  print_matrix(r_mat, NROW, NCOL);
  // free gsl_rng and return
  gsl_rng_free(rng);
#else
  printf("\nGSL disabled: -DGSL_INCLUDE not specified\n\n");
#endif /* GSL_INCLUDE */
#endif /* __INTELLISENSE__ */

  return EXIT_SUCCESS;
}