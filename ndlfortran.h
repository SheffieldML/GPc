/*This file contains interfaces to various useful FORTRAN functions which are called in the code. Many of the FORTRAN functions are from ACM transactions, others are from the book Computation of Special Functions by Zhang and Jin. See copyright messages below and in the FORTRAN code for further information. The ACM Transactions have the associated algorithm numbers. Also provided are various erf implementations by William J. Cody.

*/
#ifndef NDLFORTRAN_H
#define NDLFORTRAN_H

// This file contains header information for various fortran functions
// that are used.

/********************** From the book *****************************/
// http://jin.ece.uiuc.edu/routines/routines.html
// 		*   COMPUTATION OF SPECIAL FUNCTIONS   *
// 		*                                      *
// 		*   Shanjie Zhang and Jianming Jin     *
// 		*                                      *
// 		*   Copyright 1996 by John Wiley &     *
// 		*              Sons, Inc.              *
//
//      All the programs and subroutines contained in this diskette
// are copyrighted.  However, we give permission to the reader who
// purchases this book to incorporate any of these programs into his
// or her programs provided that the copyright is acknowledged.

// random number generator (for setting the seed). 
extern "C" void srand_(const int& m);
extern "C" double rand_(const int& m);

// computes the psi function putting the answer in y.
extern "C" void psi_(const double& x, // input value
		      double& y);      // output
// computes the gamma function.
extern "C" void lgama_(const int& type, // 1 for gamma 0 for log gamma.
		       const double& x, // input value.
		       double& y);      // output.


/********************** William Cody ****************************/
// This is William Cody's scaled complementary erf implementation 
extern "C" double derfcx_(const double& x);

/********************** ACM Algorithms **************************/
// this is algorithm 488 for GRAND, Gaussian sampling.
extern "C" double dgrand_(const int& N); // N is a dummy parameter

// Algorithm 513 (revised 380): in-place transpose of an M-by-N matrix
// stored column-wise in A[0..MN-1]. Recommended lwork is (M+N)/2.
// Argument order matches Fortran DTRANSR(A, M, N, MN, MOVE, IWRK, IOK).
extern "C" void dtransr_(double* A,
		       const int& M,   // rows before transpose
		       const int& N,   // cols before transpose
		       const int& MN,  // M*N
		       int* work,
		       const int& lwork,
		       int& iok);

// this is l-bfgs from http://www.ece.northwestern.edu/%7Enocedal/lbfgs.html
extern "C" void lbfgs_(const int& numVariables, 
		       const int& numCorrections,
		       double* X,
		       const double& funcVal,   // set by user to be func val.
		       const double* gradVals,  // set by user to be grad vals.
		       const int& diagCo,
		       const double* diag,
		       const int iPrint[2],
		       const double& prec,
		       const double& xtol,
		       double* W, // work vector size N(2M+1) + 2M
		       int& iFlag);


#endif
