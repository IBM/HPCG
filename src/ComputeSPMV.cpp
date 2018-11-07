
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
#include <cstdio>

#if defined(__IBMC__) || defined(__IBMCPP__)
#include "ibm.hpp"
#endif

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y)
{
  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo_opt(A,x);
#endif

  // Constants
  const local_int_t nrow = A.localNumberOfRows;

#if defined(__PPC64__)
  __attribute__ ((__aligned__ (16))) const double* const xv = x.values;
  __attribute__ ((__aligned__ (16))) double * const yv = y.values;
#else
  const double * const xv = x.values;
  double * const yv = y.values;
#endif

  register double sum0, sum1;
  register local_int_t id0, id1;
  register local_int_t i, j;
  register local_int_t rowStart;
  register local_int_t rowEnd;

#ifndef HPCG_NO_OPENMP
    register local_int_t tID;
    #pragma omp parallel private (i,j,sum0,sum1,id0,id1,rowStart,rowEnd,tID)
    {
      tID = omp_get_thread_num();
      rowStart = A.optimizationData[tID][0];
      rowEnd   = A.optimizationData[tID][1];
#else
      rowStart = 0;
      rowEnd   = nrow-1;
#endif

#if defined (__HAVE_ELLPACK_FORMAT)
      const double * Av = &A.optimizedEllpackVals[0];
      const local_int_t * Ac = &A.optimizedEllpackCols[0];
#else
      const double * Av = A.matrixValues[0];
      const local_int_t * Ac = A.mtxIndL[0];
#endif

#pragma disjoint (*Av, *Ac, *xv, *yv)

      //#pragma ibm independent_loop
      for ( i = rowStart; i <= rowEnd; i+=2)
      {
#if defined (__HAVE_ELLPACK_FORMAT)
        id0  = i     * ELLPACK_SIZE;
        id1  = (i+1) * ELLPACK_SIZE;
#else
        id0  = i     * MAX_ELEMENTS_PER_ROW;
        id1  = (i+1) * MAX_ELEMENTS_PER_ROW;
#endif
        sum0 = 0.0;
        sum1 = 0.0;
        for ( j = 0; j < MAX_ELEMENTS_PER_ROW; ++j )
        {
          sum0 += Av[id0+j] * xv[Ac[id0+j]];
          sum1 += Av[id1+j] * xv[Ac[id1+j]];
        }
        yv[i]   = sum0;
        yv[i+1] = sum1;
      }
#ifndef HPCG_NO_OPENMP
    }
#endif
  return 0;
}
