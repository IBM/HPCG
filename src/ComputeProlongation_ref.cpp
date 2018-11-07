
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
 @file ComputeProlongation_ref.cpp

 HPCG routine
 */

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeProlongation_ref.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[in]  Af - Fine grid sparse matrix object containing pointers to current coarse grid correction and the f2c operator.
  @param[inout] xf - Fine grid solution vector, update with coarse grid correction.

  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeProlongation_ref(const SparseMatrix & Af, Vector & xf) {

  double * xfv = xf.values;
  double * xcv = Af.mgData->xc->values;
  local_int_t * f2c = Af.mgData->f2cOperator;
  local_int_t nc = Af.mgData->rc->localLength;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
// TODO: Somehow note that this loop can be safely vectorized since f2c has no repeated indices
  for (local_int_t i=0; i<nc; ++i) xfv[f2c[i]] += xcv[i]; // This loop is safe to vectorize

  return 0;
}

int ComputeProlongation(const SparseMatrix & Af, Vector & xf) {

//#if !defined(__IBMC__) && !defined(__IBMCPP__)

//  return(ComputeProlongation_ref(Af, xf));

//#else

  double * xfv = xf.values;
  double * xcv = Af.mgData->xc->values;
  local_int_t * f2c = Af.mgData->f2cOperator;

#pragma disjoint (*xfv, *xcv, *f2c)

  register local_int_t j;
  register local_int_t jStart;
  register local_int_t jEnd;
  register local_int_t tID;

  #pragma omp parallel private (j,jStart,jEnd,tID)
  {
    tID = omp_get_thread_num();
    jStart = Af.Ac->optimizationData[tID][0];
    jEnd   = Af.Ac->optimizationData[tID][1];

#if defined(__IBMC__) || defined(__IBMCPP__)
//  #pragma ibm independent_loop
#endif
    for (j = jStart; j <= jEnd; ++j)
        xfv[f2c[j]] += xcv[j];
  }

  return 0;
//#endif
}
