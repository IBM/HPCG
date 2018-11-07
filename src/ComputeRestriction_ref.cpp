
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

/*!
 @file ComputeRestriction_ref.cpp

 HPCG routine
 */


#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeRestriction_ref.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeRestriction_ref(const SparseMatrix & A, const Vector & rf) {

  double * Axfv = A.mgData->Axf->values;
  double * rfv = rf.values;
  double * rcv = A.mgData->rc->values;
  local_int_t * f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->localLength;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for (local_int_t i=0; i<nc; ++i) rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];

  return 0;
}

int ComputeRestriction(const SparseMatrix & A, const Vector & rf) {

#if !defined(__IBMC__) && !defined(__IBMCPP__)

  return(ComputeRestriction_ref(A, rf));

#else

  double * Axfv = A.mgData->Axf->values;
  double * rfv = rf.values;
  double * rcv = A.mgData->rc->values;
  local_int_t * f2c = A.mgData->f2cOperator;

#pragma disjoint (*Axfv, *rfv, *rcv, *f2c)

  register local_int_t j;
  register local_int_t jStart;
  register local_int_t jEnd;
  register local_int_t tID;

  #pragma omp parallel private (j,jStart,jEnd,tID)
  {
    tID = omp_get_thread_num();
//    jStart = rf.optimizationData[tID][0];
//    jEnd   = rf.optimizationData[tID][1];
   
    jStart = A.Ac->optimizationData[tID][0];
    jEnd   = A.Ac->optimizationData[tID][1];
 

#if defined(__IBMC__) || defined(__IBMCPP__)
//  #pragma ibm independent_loop
#endif
    for (j = jStart; j <= jEnd; ++j)
        rcv[j] = rfv[f2c[j]] - Axfv[f2c[j]];
  }

  return 0;
#endif
}
