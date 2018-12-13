
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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
#include <cstdio>

#if defined(__IBMC__) || defined(__IBMCPP__)
#include "ibm.hpp"
#endif

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {
#if !defined(__IBMC__) && !defined(__IBMCPP__)

  isOptimized = false;
  return ComputeWAXPBY_ref(n, alpha, x, beta, y, w);

#else

  assert(x.localLength>=n);
  assert(y.localLength>=n);
  assert(w.localLength>=n);

  double * xv = x.values;
  double * yv = y.values;
  double * wv = w.values;

  register local_int_t j;
  register local_int_t jStart;
  register local_int_t jEnd;
  register local_int_t tID;

  // THIS IS THE ONLY CASE USED IN THIS BENCHMARK
  if ( alpha == 1.0 )
  {
    // MOSTLY USING THIS CASE
    if ( beta != 0.0 )
    {
#if defined(__bgq__)
      const vector4double b_vd = vec_splats(beta);
#elif defined(__PPC64__)
      const vector double b_vd = vec_splats(beta);
#endif
      if (xv == wv)
      {
        #pragma omp parallel private (j,jStart,jEnd,tID)
        {
          tID = omp_get_thread_num();
          jStart = x.optimizationData[tID][0];
          jEnd   = x.optimizationData[tID][1];

          // NOTE: we check a priori during the optimization that n is divisible by 4!
#if defined(__bgq__)
          for (j = jStart; j <= jEnd; j+=4)
            vec_st(vec_madd(vec_ld(0, &yv[j]), b_vd, vec_ld(0, &xv[j])), 0, &xv[j]);
#elif defined(__PPC64__)
          for (j = jStart; j <= jEnd; j+=2)
            vec_xst(vec_madd(*(vector double*) &yv[j], b_vd, *(vector double*) &xv[j]), 0, &xv[j]);
#endif
        }
        return 0;
      }
      else if (yv == wv)
      {
        #pragma omp parallel private (j,jStart,jEnd,tID)
        {
          tID = omp_get_thread_num();
          jStart = x.optimizationData[tID][0];
          jEnd   = x.optimizationData[tID][1];

          // NOTE: we check a priori during the optimization that n is divisible by 4!
#if defined(__bgq__)
          for (j = jStart; j <= jEnd; j+=4)
            vec_st(vec_madd(vec_ld(0, &yv[j]), b_vd, vec_ld(0, &xv[j])), 0, &yv[j]);
#elif defined(__PPC64__)
          for (j = jStart; j <= jEnd; j+=2)
            vec_xst(vec_madd(*(vector double*) &yv[j], b_vd, *(vector double*) &xv[j]), 0, &yv[j]);
#endif
        }
        return 0;
      }
      else
      {
        #pragma omp parallel private (j,jStart,jEnd,tID)
        {
          tID = omp_get_thread_num();
          jStart = x.optimizationData[tID][0];
          jEnd   = x.optimizationData[tID][1];

          // NOTE: we check a priori during the optimization that n is divisible by 4!
#if defined(__bgq__)
          for (j = jStart; j <= jEnd; j+=4)
            vec_st(vec_madd(vec_ld(0, &yv[j]), b_vd, vec_ld(0, &xv[j])), 0, &wv[j]);
#elif defined(__PPC64__)
          for (j = jStart; j <= jEnd; j+=2)
            vec_xst(vec_madd(*(vector double*) &yv[j], b_vd, *(vector double*) &xv[j]), 0, &wv[j]);
#endif
        }
        return 0;
      }
    }
    else
    {
      // BASICALLY THIS IS A COPY - STRANGE THEY DO IT WITH AXPY

      #pragma omp parallel private (j,jStart,jEnd,tID)
      {
        tID = omp_get_thread_num();
        jStart = x.optimizationData[tID][0];
        jEnd   = x.optimizationData[tID][1];

        // NOTE: we check a priori during the optimization that n is divisible by 4!
#if defined(__bgq__)
        for (j = jStart; j <= jEnd; j+=4)
          vec_st(vec_ld(0, &xv[j]), 0, &wv[j]);
#elif defined(__PPC64__)
        for (j = jStart; j <= jEnd; j+=2)
          vec_xst(*(vector double*) &xv[j], 0, &wv[j]);
#endif
      }
      return 0;
    }
  }
  else
  {
    // TODO - FINISH IMPLEMENTATION - NOT NEEDED IN THIS BENCHMARK
    printf("WAXPBY: Alpha != 1.0: %22.16e\n", alpha);

    return -1;
  }
  return 0;

#endif

}
