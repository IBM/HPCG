
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"

#if defined(__IBMC__) || defined(__IBMCPP__)
#include "ibm.hpp"
//#include "essl.h"
#endif

// #define ddot

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

#if !defined(__IBMC__) && !defined(__IBMCPP__)

  isOptimized = false;
  return ComputeDotProduct_ref(n, x, y, result, time_allreduce);

#elif 0 //ESSL IMPLEMENTATION

  const register int IONE = 1;
  double local_result = ddot(n, x.values, IONE, y.values, IONE);

// This part is the same as in the REF version
#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  time_allreduce -= mytimer();
  result = 0.0;
  MPI_Allreduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  time_allreduce += mytimer();
#else
  result = local_result;
#endif

  return(0);

#else

  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  double * xv = x.values;
  double * yv = y.values;
  double local_result = 0.0;

  register local_int_t tID;
  register local_int_t j;
  register local_int_t jStart;
  register local_int_t jEnd;

  if (yv==xv)
  {
    #pragma omp parallel reduction(+:local_result) private (j,jStart,jEnd,tID)
    {
      tID = omp_get_thread_num();
      jStart = x.optimizationData[tID][0];
      jEnd   = x.optimizationData[tID][1];

#if defined(__bgq__)
      vector4double sum_vd = vec_splats(0.0);
      vector4double x_vd;
#elif defined(__PPC64__)
      vector double sum_vd = vec_splats(0.0);
      vector double x_vd;
      double *psum;
#endif

      // NOTE: we check a priori during the optimization that n is divisible by 4!
#if defined(__bgq__)
      for (j = jStart; j <= jEnd; j+=4)
#elif defined(__PPC64__)
      for (j = jStart; j <= jEnd; j+=2)
#endif
      {
#if defined(__bgq__)
        x_vd = vec_ld(0, &xv[j]);
#elif defined(__PPC64__)
        x_vd = *(vector double*) &xv[j];
#endif
        sum_vd = vec_madd(x_vd,x_vd,sum_vd);
      }
#if defined(__bgq__)
      x_vd = vec_add(vec_perm(sum_vd,sum_vd,vec_gpci(03210)),sum_vd);
      local_result = vec_extract(vec_add(x_vd,vec_perm(x_vd,x_vd,vec_gpci(02301))), 0);
      //local_result = sum_vd[0] + sum_vd[1] + sum_vd[2] + sum_vd[3];
#elif defined(__PPC64__)
      psum = (double *) &sum_vd;
      local_result  = psum[0] + psum[1];
      //local_result = sum_vd[0] + sum_vd[1];
#endif
    }
  }
  else
  {
    #pragma omp parallel reduction(+:local_result) private (j,jStart,jEnd,tID)
    {
      tID = omp_get_thread_num();
      jStart = x.optimizationData[tID][0];
      jEnd   = x.optimizationData[tID][1];

#if defined(__bgq__)
      vector4double sum_vd = vec_splats(0.0);
#elif defined(__PPC64__)
      vector double sum_vd = vec_splats(0.0);
      double *psum;
#endif
      // NOTE: we check a priori during the optimization that n is divisible by 4!
#if defined(__bgq__)
      for (j = jStart; j <= jEnd; j+=4)
        sum_vd = vec_madd(vec_ld(0, &xv[j]),vec_ld(0, &yv[j]),sum_vd);
      vector4double x_vd = vec_add(vec_perm(sum_vd,sum_vd,vec_gpci(03210)),sum_vd);
      local_result = vec_extract(vec_add(x_vd,vec_perm(x_vd,x_vd,vec_gpci(02301))), 0);
//      local_result = sum_vd[0] + sum_vd[1] + sum_vd[2] + sum_vd[3];
#elif defined(__PPC64__)
      for (j = jStart; j <= jEnd; j+=2)
        sum_vd = vec_madd(*(vector double*) &xv[j], *(vector double*) &yv[j],sum_vd);
      psum = (double *) &sum_vd;
      local_result  = psum[0] + psum[1];
      //local_result = sum_vd[0] + sum_vd[1];
#endif
    }
  }

// This part is the same as in the REF version
#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  time_allreduce -= mytimer();
  result = 0.0;
  MPI_Allreduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  time_allreduce += mytimer();
#else
  result = local_result;
#endif

  return 0;

#endif
}
