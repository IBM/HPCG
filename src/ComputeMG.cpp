
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
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"
#include <cassert>
#include <iostream>

#if defined(__IBMC__) || defined(__IBMCPP__)
#include "ibm.hpp"
#endif

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG
*/
int ComputeMG(const SparseMatrix & A, const Vector & r, Vector & x) {

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
#if defined(__HAVE_HPM)
    HPM_Start("MG_PreSmoother");
#endif
    for (int i=0; i< numberOfPresmootherSteps; ++i)
    {
        // If only one thread use simple implementation
        if (A.optimizationNmaxThreads == 1 || A.optimizationData == 0)
            ierr += ComputeSYMGS_noThreads(A, r, x);
        else
#if defined(__HAVE_ASYNC_SYMGS)
            ierr += ComputeSYMGS_Async(A, r, x);
#else
            ierr += ComputeSYMGS(A, r, x);
#endif
    }
#if defined(__HAVE_HPM)
    HPM_Stop("MG_PreSmoother");
#endif
    if (ierr!=0) return(ierr);
    ierr = ComputeSPMV(A, x, *A.mgData->Axf); if (ierr!=0) return(ierr);
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction(A, r);  if (ierr!=0) return(ierr);
    ierr = ComputeMG(*A.Ac,*A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return(ierr);
    ierr = ComputeProlongation(A, x);  if (ierr!=0) return(ierr);
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i)
    {
        // If only one thread use simple implementation
        if (A.optimizationNmaxThreads == 1 || A.optimizationData == 0)
            ierr += ComputeSYMGS_noThreads(A, r, x);
        else
#if defined(__HAVE_ASYNC_SYMGS)
            ierr += ComputeSYMGS_Async(A, r, x);
#else
            ierr += ComputeSYMGS(A, r, x);
#endif
    }
    if (ierr!=0) return(ierr);
  }
  else {
#if defined(__HAVE_HPM)
    HPM_Start("MG_SYMGS");
#endif
    ierr = ComputeSYMGS(A, r, x);
#if defined(__HAVE_HPM)
    HPM_Stop("MG_SYMGS");
#endif
    if (ierr!=0) return(ierr);
  }
  return 0;
}

