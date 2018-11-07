
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

#ifndef COMPUTESYMGS_HPP
#define COMPUTESYMGS_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

void ComputeSYMGS_forwardStep ( const int& start, const int& stop, const SparseMatrix& A, const double* const rv, double * const xv);
void ComputeSYMGS_backwardStep( const int& start, const int& stop, const SparseMatrix& A, const double* const rv, double * const xv);
int ComputeSYMGS_noThreads( const SparseMatrix & A, const Vector & r, Vector & x);
int ComputeSYMGS_Async( const SparseMatrix & A, const Vector & r, Vector & x);
int ComputeSYMGS( const SparseMatrix  & A, const Vector & r, Vector & x);

#endif // COMPUTESYMGS_HPP
