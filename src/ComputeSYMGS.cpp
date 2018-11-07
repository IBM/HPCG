
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
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"

#if defined(__IBMC__) || defined(__IBMCPP__)
#include "ibm.hpp"
#endif

// TODO - REMOVE THESE ARE JUST FOR DEBUGGING
#include <cstdio>
#include <cstdlib>

#include "hpcg.hpp"

/*!
  Routine to compute one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
  - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSYMGS_ref
*/

// Use ref method - for testing
/*
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {
   return(ComputeSYMGS_ref(A, r, x));
}
*/

/*
#if defined(__bgq__)
static int prefetcherCounter = 0;
#endif
*/

void ComputeSYMGS_forwardStep( const int& start, const int& stop, const SparseMatrix& A, const double* const rv, double * const xv)
{
    register double sum = 0.;

#if 0
    for (register local_int_t i = start; i <= stop; ++i)
    {

      // Define variables //TODO Clean this part and make it faster
      const double * const currentValues = A.matrixValues[i];
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];
      const double  currentDiagonal = A.matrixDiagonal[i][0]; // Current diagonal value

      // RHS value
      sum = rv[i];

      // Add all elements
      for (register int j = 0; j < currentNumberOfNonzeros; ++j)
        sum -= currentValues[j] * xv[currentColIndices[j]];

      // Remove diagonal contribution from previous loop
      sum += currentDiagonal * xv[i];

      // Divide by the diagonal
      xv[i] = sum / currentDiagonal;
    }
#else
#if defined (__HAVE_ELLPACK_FORMAT)
    const double * Av = &A.optimizedEllpackVals[start*ELLPACK_SIZE];
    const local_int_t * Ac = &A.optimizedEllpackCols[start*ELLPACK_SIZE];
    for (register local_int_t i = start; i <= stop; ++i, Av += ELLPACK_SIZE, Ac += ELLPACK_SIZE)
    {
#if defined(__IBMC__) || defined(__IBMCPP__)
      __dcbt( (void *)&Av[ELLPACK_SIZE] );
      __dcbt( (void *)&Ac[ELLPACK_SIZE] );
#endif
      for ( register int j = 1; j < ELLPACK_SIZE-1; ++j)
        sum -= Av[j] * xv[Ac[j]];

#if defined(__HAVE_ELLPACK_FORMAT_WITHOUTDIAG)
      xv[i] = (rv[i] + sum) / Av[0];
#else
      xv[i] = (rv[i] + sum) * Av[ELLPACK_SIZE-1];
#endif
      sum = 0.;
    }
#else
    const double * Av = A.matrixValues[start];
    const local_int_t * Ac = A.mtxIndL[start];
    double ** Ad = &A.matrixDiagonal[start];

    for (register local_int_t i = start; i <= stop; ++i, Av += MAX_ELEMENTS_PER_ROW, Ac += MAX_ELEMENTS_PER_ROW, ++Ad)
    {
#if defined(__IBMC__) || defined(__IBMCPP__)
      __dcbt( (void *)&Av[MAX_ELEMENTS_PER_ROW] );
      __dcbt( (void *)&Ac[MAX_ELEMENTS_PER_ROW] );
#endif
      for ( register int j = 0; j < MAX_ELEMENTS_PER_ROW; ++j)
        sum -= Av[j] * xv[Ac[j]];

      xv[i] = (rv[i] + sum + *Ad[0] * xv[i]) / *Ad[0];
      sum = 0.;
    }
#endif
#endif
}

void ComputeSYMGS_backwardStep( const int& start, const int& stop, const SparseMatrix& A, const double* const rv, double * const xv)
{
    register double sum = 0.;

#if 0
    for (register local_int_t i = start; i >= stop; --i)
    {
      // Define variables //TODO Clean this part and make it faster
      const double * const currentValues = A.matrixValues[i];
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];
      const double  currentDiagonal = A.matrixDiagonal[i][0]; // Current diagonal value

      // RHS value
      sum = rv[i];

      // Add all elements
      for (register int j = 0; j < currentNumberOfNonzeros; ++j)
        sum -= currentValues[j] * xv[currentColIndices[j]];

      // Remove diagonal contribution from previous loop
      sum += currentDiagonal * xv[i];

      // Divide by the diagonal
      xv[i] = sum / currentDiagonal;
    }
#else
#if defined (__HAVE_ELLPACK_FORMAT)
    const double * Av = &A.optimizedEllpackVals[start*ELLPACK_SIZE];
    const local_int_t * Ac = &A.optimizedEllpackCols[start*ELLPACK_SIZE];

    for (register local_int_t i = start; i >= stop; --i, Av -= ELLPACK_SIZE, Ac -= ELLPACK_SIZE)
    {
#if defined(__IBMC__) || defined(__IBMCPP__)
      __dcbt( (void *)&Av[-ELLPACK_SIZE] );
      __dcbt( (void *)&Ac[-ELLPACK_SIZE] );
#endif
      for ( register int j =  ELLPACK_SIZE-2; j > 0 ; --j)
        sum -= Av[j] * xv[Ac[j]];

#if defined(__HAVE_ELLPACK_FORMAT_WITHOUTDIAG)
      xv[i] = (rv[i] + sum) / Av[0];
#else
      xv[i] = (rv[i] + sum) * Av[ELLPACK_SIZE-1];
#endif
      sum = 0.;
    }
#else
    const double * Av = A.matrixValues[start];
    const local_int_t * Ac = A.mtxIndL[start];
    double ** Ad = &A.matrixDiagonal[start];

    for (register local_int_t i = start; i >= stop; --i, Av -= MAX_ELEMENTS_PER_ROW, Ac -= MAX_ELEMENTS_PER_ROW, --Ad)
    {
#if defined(__IBMC__) || defined(__IBMCPP__)
      __dcbt( (void *)&Av[-MAX_ELEMENTS_PER_ROW] );
      __dcbt( (void *)&Ac[-MAX_ELEMENTS_PER_ROW] );
#endif
      for ( register int j = MAX_ELEMENTS_PER_ROW-1; j > -1 ; --j)
        sum -= Av[j] * xv[Ac[j]];

      xv[i] = (rv[i] + sum + *Ad[0] * xv[i]) / *Ad[0];
      sum = 0.;
    }
#endif
#endif
}

// Does not use threads in the SYMGS (BUT USE PIVOTING)
int ComputeSYMGS_noThreads( const SparseMatrix & A, const Vector & r, Vector & x)
{
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x);
#endif

#if defined(__PPC64__)
  __attribute__ ((__aligned__ (16))) const double* const rv = r.values;
  __attribute__ ((__aligned__ (16))) double * const xv = x.values;
#else
  const double* const rv = r.values;
  double * const xv = x.values;
#endif

#pragma disjoint (*xv, *rv)
#if defined(__bgq__)
__alignx(32,xv);
__alignx(32,rv);
#endif

  // Forward-Sweep 1) compute dependency for next thread, with no extra-diag dependency from previous thread
  ComputeSYMGS_forwardStep(0, A.localNumberOfRows-1, A, rv,xv);

  // Backward-Sweep 1) compute dependency for previous thread, with no extra-diag dependency from subsequent thread
  ComputeSYMGS_backwardStep(A.localNumberOfRows-1, 0, A, rv, xv);

  return 0;
}

int ComputeSYMGS_Async( const SparseMatrix & A, const Vector & r, Vector & x)
{
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo_opt(A,x);
#endif

  // Pre-processing for threads (these variables need to be shared)
  bool * forward_sweep_stop  = A.forward_sweep_stop;
  bool * backward_sweep_stop = A.backward_sweep_stop;

  //double ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues
#if defined(__PPC64__)
  __attribute__ ((__aligned__ (16))) const double* const rv = r.values;
  __attribute__ ((__aligned__ (16))) double * const xv = x.values;
#else
  const double* const rv = r.values;
  double * const xv = x.values;
#endif

#pragma disjoint (*xv, *rv)
#if defined(__bgq__)
//L1P_SetStreamPolicy(L1P_stream_optimistic);
__alignx(32,xv);
__alignx(32,rv);
#endif

  #pragma omp parallel
  {

    // ThreadID
    int t = omp_get_thread_num();

    // Forward-Sweep 1) compute dependency for next thread, with no extra-diag dependency from previous thread
    ComputeSYMGS_forwardStep(A.optimizationData[t][2], A.optimizationData[t][7], A, rv,xv);

    // Barrier before backward-sweep
    #pragma omp barrier

    // Backward-Sweep 1) compute dependency for previous thread, with no extra-diag dependency from subsequent thread
    ComputeSYMGS_backwardStep(A.optimizationData[t][8], A.optimizationData[t][13], A, rv,xv);

  }

  return 0;
}


int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x)
{
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo_opt(A,x);
#endif

  // Pre-processing for threads (these variables need to be shared)
  bool * forward_sweep_stop  = A.forward_sweep_stop;
  bool * backward_sweep_stop = A.backward_sweep_stop;

  //double ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues
#if defined(__PPC64__)
  __attribute__ ((__aligned__ (16))) const double* const rv = r.values;
  __attribute__ ((__aligned__ (16))) double * const xv = x.values;
#else
  const double* const rv = r.values;
  double * const xv = x.values;
#endif

#pragma disjoint (*xv, *rv)
#if defined(__bgq__)
//L1P_SetStreamPolicy(L1P_stream_optimistic);
__alignx(32,xv);
__alignx(32,rv);
#endif

  #pragma omp parallel
  {
/*
#if defined(__bgq__)
    if(prefetcherCounter)
      L1P_PatternStart(0);
    else
    {
      L1P_PatternConfigure(1000000);
      L1P_PatternStart(1);
    }
#endif
*/

    // ThreadID
    int t = omp_get_thread_num();



    // Forward-Sweep 1) compute dependency for next thread, with no extra-diag dependency from previous thread
    ComputeSYMGS_forwardStep(A.optimizationData[t][2], A.optimizationData[t][3], A, rv,xv);

    // Free lock for next thread;
    backward_sweep_stop[t] = true;
    forward_sweep_stop[t] = false;

    // Forward-Sweep 2) compute remaining elements with no extra-diag dependency from previous thread
    ComputeSYMGS_forwardStep(A.optimizationData[t][4], A.optimizationData[t][5], A, rv,xv);

    if (t > 0)
    {
      // Wait until the previous thread has computed the extra-diag dependencies
      while (forward_sweep_stop[t-A.optimizationDataDelta[t]])
      {
        #pragma omp flush(forward_sweep_stop)
      }

      // Forward-Sweep 3) compute remaining elements with extra-diag dependency from previous thread
      ComputeSYMGS_forwardStep(A.optimizationData[t][6], A.optimizationData[t][7], A, rv,xv);
    }



    // Barrier before backward-sweep
    #pragma omp barrier



    // Backward-Sweep 1) compute dependency for previous thread, with no extra-diag dependency from subsequent thread
    ComputeSYMGS_backwardStep(A.optimizationData[t][8], A.optimizationData[t][9], A, rv,xv);

    // Free lock for previous thread;
    forward_sweep_stop[t] = true;
    backward_sweep_stop[t] = false;

    // Backward-Sweep 2) compute remaining elements with no extra-diag dependency from subsequent thread
    ComputeSYMGS_backwardStep(A.optimizationData[t][10], A.optimizationData[t][11], A, rv,xv);

    if (t < (A.optimizationNmaxThreads - A.optimizationDataDelta[t]))
    {
      // Wait until the previous thread has computed the extra-diag dependencies
      while (backward_sweep_stop[t+A.optimizationDataDelta[t]])
      {
        #pragma omp flush(backward_sweep_stop)
      }

      // Backward-Sweep 3) compute remaining elements with extra-diag dependency from subsequent thread
      ComputeSYMGS_backwardStep(A.optimizationData[t][12], A.optimizationData[t][13], A, rv,xv);
    }
/*
#if defined(__bgq__)
L1P_SetStreamPolicy(L1P_confirmed_or_dcbt);
#endif
*/

/*
#if defined(__bgq__)
    ++prefetcherCounter;
    L1P_PatternStop();
    if (prefetcherCounter % 100 == 0)
    {
      prefetcherCounter = 0;
      L1P_PatternUnconfigure();
    }
#endif
*/
  }

  return 0;
}




#if defined(YVES_OLD_IMPLEMENTATION)
#if defined(__bgq__)
//XXX: using a static var to learn the pattern only in the first call to
//     ComputeSYMGS
static int count = 0;
//static std::vector<local_int_t> diagonal_index(1, 0);
//std::vector<double> upper_triangle_sum(1, 0.0);
//std::vector<double> col_vec(1000, 0.0);
#endif

int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {


  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  const local_int_t nrow = A.localNumberOfRows;
  double ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues
  const double * const rv = r.values;
  double * const xv = x.values;

#if defined(__bgq__)
#pragma omp parallel
{
    if(count == 0)
        L1P_PatternConfigure(1000000);

    if(count == 0)
        L1P_PatternStart(1);
    else
        L1P_PatternStart(0);
}
#endif

#if 0
  // in the first call compute the index of the diagonal entry (column index)
  // or the first off diagonal entry in the upper triangular part.
  if(count == 0) {
    diagonal_index.resize(nrow);
    upper_triangle_sum.resize(nrow);
#pragma omp parallel for
    for (local_int_t i=0; i < nrow; i++) {
        const local_int_t * const currentColIndices = A.mtxIndL[i];

        int j = 0;
        while(j < A.nonzerosInRow[i] && currentColIndices[j] < i) { j++; }
        diagonal_index[i] = j;
    }
  }


HPM_Start("SYMGS_INNER");

  // first we can compute the strictly upper triangular part of the matvec
  // in parallel
#pragma omp parallel for schedule(static, 64)
  for (local_int_t i=0; i < nrow; i++) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];

    int j = diagonal_index[i];
    // make sure not to start on the diagonal
    if(currentColIndices[j] == i) j++;

    double sum = 0.0;

    for (; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum += currentValues[j] * xv[curCol];
    }

    upper_triangle_sum[i] = sum;
  }


  // now we do the serial update
  for (local_int_t i=0; i< nrow; i++) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const double  currentDiagonal = matrixDiagonal[i][0];

    double sum = rv[i]; // RHS value

    // the last index is either the last entry before the diagonal
    // or the diagonal entry itself (diagonal_index is either diagonal entry
    // or first entry on upper diagonal matrix)
    int end_index = diagonal_index[i] - 1;
    // make sure not to start on the diagonal
    if(currentColIndices[end_index] == i) end_index--;

    for (int j=0; j < end_index; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }

    sum -= upper_triangle_sum[i];

    // remove diagonal if we incorporated it
    //if(currentColIndices[end_index] == i)
        //sum += xv[i] * currentDiagonal;

    xv[i] = sum / currentDiagonal;
  }
#endif


  double * col_vec; //, *col_vec_b;
  col_vec = (double*)malloc(1000*sizeof(double));

  for (local_int_t i=0; i< nrow; i++) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value

    for (int j=0; j< currentNumberOfNonzeros; j++){
      local_int_t curCol = currentColIndices[j];
      col_vec[j] = xv[curCol];
    }


    //#pragma omp parallel for reduction ( + : sum)
#if defined(__bgq__)
    __alignx(16,currentValues);
    __alignx(16,col_vec);
#endif
    for (int j=0; j< currentNumberOfNonzeros; j++) {
      //local_int_t curCol = currentColIndices[j];
      //sum -= currentValues[j] * xv[curCol];
      sum -= currentValues[j] * col_vec[j];
    }
    sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum/currentDiagonal;
  }

//HPM_Stop("SYMGS_INNER");


  // Now the back sweep.

  for (local_int_t i=nrow-1; i>=0; i--) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value

    for (int j = 0; j< currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j]*xv[curCol];
    }
    sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum/currentDiagonal;
  }

#if defined(__bgq__)
#pragma omp parallel
{
    L1P_PatternStop();

    //L1P_PatternStatus(&st);
    //HPCG_fout << "maxed = "<< st.s.maximum << ", abandoned = " <<  st.s.abandoned << '\n';
}
#endif

  free(col_vec);
  count++;

  return(0);
}
#endif
