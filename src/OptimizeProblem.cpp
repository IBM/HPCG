
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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include "OptimizeProblem.hpp"

#include <utility>
#include <functional>
#include <algorithm>
#include <vector>

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include <cstdio>
#include <cstdlib>

#include <mpi.h>
#include "mytimer.hpp"
#include "ibm.hpp"

bool pairComparator ( const local_int_t& l, const local_int_t& r)
{
  return l < r;
}

int InitializeMatrixOptimizationData(SparseMatrix& A, const int& n_threads, const int& n_max_threads)
{
  // Create matrix structure for SymGS thread optimization
  // This must be available to all threads: out of parallel region
  // We need the first 2 values also in case only 1 thread is available
  A.optimizationData = new local_int_t*[n_max_threads];
  A.optimizationDataDelta = new local_int_t[n_max_threads];
  allocatedMemory2Optimize += ( n_max_threads + n_max_threads * 14 ) * sizeof (local_int_t);

  const int nrows = A.localNumberOfRows;
  if ( nrows % n_threads > 0 )
  {
    printf("!!! ERROR - Pivoting not possible: nrows: %d, n_threads: %d\n",nrows, n_threads);
    exit(-1);
  }
  const int block_size = nrows / n_threads;
  for (int i = 0, j = 0; i < n_max_threads; ++i)
  {
    A.optimizationData[i]      = new local_int_t[14]();
    A.optimizationDataDelta[i] = n_max_threads / n_threads;

    if ( i % (A.optimizationDataDelta[i]) == 0)
    {
      if (n_threads == 1)
      {
        // Thread block
        A.optimizationData[i][0] = block_size * j;           // First row in block
        A.optimizationData[i][1] = block_size * (j + 1) - 1; // Last row in block

        // 1st Forward
        A.optimizationData[i][2]  =  0;
        A.optimizationData[i][3]  = -1;

        // 2nd Forward
        A.optimizationData[i][4]  =  A.optimizationData[i][0];
        A.optimizationData[i][5]  =  A.optimizationData[i][1];

        // 3rd Forward
        A.optimizationData[i][6]  =  0;
        A.optimizationData[i][7]  = -1;

        // 1st Backward
        A.optimizationData[i][8]  = -1;
        A.optimizationData[i][9]  =  0;

        // 2nd Backward
        A.optimizationData[i][10] =  A.optimizationData[i][1];
        A.optimizationData[i][11] =  A.optimizationData[i][0];

        // 3rd Backward
        A.optimizationData[i][12] = -1;
        A.optimizationData[i][13] =  0;
      }
      else
      {
        // Thread block
        A.optimizationData[i][0] = block_size * j;           // First row in block
        A.optimizationData[i][1] = block_size * (j + 1) - 1; // Last row in block

        // 1st Forward
        A.optimizationData[i][2] =  A.optimizationData[i][0];  // OK
        A.optimizationData[i][3] =  A.optimizationData[i][0];

        // 2nd Forward
        A.optimizationData[i][4] =  A.optimizationData[i][0];  // ID of first row that does not create any extra-diag dependency for next thread
                                                               // (initialized as first row of the block)
        A.optimizationData[i][5] =  A.optimizationData[i][0];

        // 3rd Forward
        A.optimizationData[i][6] =  block_size * (j + 1);      // First row with extra-diag dependency
                                                               // (initialized as first row of next block)
        A.optimizationData[i][7] =  A.optimizationData[i][1];  // OK

        // 1st Backward
        A.optimizationData[i][8] =  A.optimizationData[i][1];  // OK
        A.optimizationData[i][9] =  A.optimizationData[i][1];

        // 2nd Backward
        A.optimizationData[i][10] =  A.optimizationData[i][1]; // ID of first row (from bottom) that does not create any extra-diag dependency for the previous thread
                                                               // (initialized as last row of the block)
        A.optimizationData[i][11] =  A.optimizationData[i][1];

        // 3rd Backward
        A.optimizationData[i][12] =  block_size * j - 1;       // First row (from bottom) with extra-diag dependency
                                                               // (initialized as last row of previous block)
        A.optimizationData[i][13] =  A.optimizationData[i][0]; // OK
      }
/*    // OLD VERSION
      // Forward-Sweep
      A.optimizationData[i][2] = block_size * (j + 1);     // First row with extra-diag dependency
                                                           // (initialized as first row of next block)
      A.optimizationData[i][3] = A.optimizationData[i][0]; // ID of first row that does not create any extra-diag dependency for next thread
                                                           // (initialized as first row of the block)

      // Backward-Sweep
      A.optimizationData[i][4] = block_size * j - 1;       // First row (from bottom) with extra-diag dependency
                                                           // (initialized as last row of previous block)
      A.optimizationData[i][5] = A.optimizationData[i][1]; // ID of first row (from bottom) that does not create any extra-diag dependency for the previous thread
                                                           // (initialized as last row of the block)
*/
      ++j;
    }
    else
    {
      A.optimizationDataDelta[i] = 0;

      // Thread block
      A.optimizationData[i][0]  =  0;
      A.optimizationData[i][1]  = -1;

      // 1st Forward
      A.optimizationData[i][2]  =  0;
      A.optimizationData[i][3]  = -1;

      // 2nd Forward
      A.optimizationData[i][4]  =  0;
      A.optimizationData[i][5]  = -1;

      // 3rd Forward
      A.optimizationData[i][6]  =  0;
      A.optimizationData[i][7]  = -1;

      // 1st Backward
      A.optimizationData[i][8]  = -1;
      A.optimizationData[i][9]  =  0;

      // 2nd Backward
      A.optimizationData[i][10] = -1;
      A.optimizationData[i][11] =  0;

      // 3rd Backward
      A.optimizationData[i][12] = -1;
      A.optimizationData[i][13] =  0;

/*
      // Forward-Sweep
      A.optimizationData[i][2] = -1;                       // First row with extra-diag dependency
                                                           // (initialized as first row of next block)
      A.optimizationData[i][3] = -1;                       // ID of first row that does not create any extra-diag dependency for next thread
                                                           // (initialized as first row of the block)

      // Backward-Sweep
      A.optimizationData[i][4] = 0;                        // First row (from bottom) with extra-diag dependency
                                                           // (initialized as last row of previous block)
      A.optimizationData[i][5] = 0;                        // ID of first row (from bottom) that does not create any extra-diag dependency for the previous thread
                                                           // (initialized as last row of the block)
*/
    }
  }

  return(0);
}

int FinalizeMatrixOptimizationData(SparseMatrix& A, const int& n_max_threads)
{
  // Create matrix structure for SymGS thread optimization
  // This must be available to all threads: out of parallel region
  // We need the first 2 values also in case only 1 thread is available
  for (int i = 0 ; i < n_max_threads; ++i)
  {
    if ( A.optimizationDataDelta[i] > 0)
    {
      // Forward
      A.optimizationData[i][3] =  A.optimizationData[i][4]-1;
      A.optimizationData[i][5] =  A.optimizationData[i][6]-1;

      // Backward
      A.optimizationData[i][9] =  A.optimizationData[i][10]+1;
      A.optimizationData[i][11] =  A.optimizationData[i][12]+1;
    }
  }

  return(0);
}

int ComputePivot(local_int_t * &pivot, const int& nrows, const int& n_threads)
{
  // Variables for permutation
  pivot = new local_int_t[nrows];

  const int block_size = nrows / n_threads;
  const int flip_block_size = block_size / 2;

  // Start 1st parallel region
  #pragma omp parallel
  {
    int tID = omp_get_thread_num();

    if (tID < n_threads)
    {
      // Compute pivoting vector
      local_int_t from0, from1, to0, to1;
      for (local_int_t j = 0 ; j < flip_block_size ; ++j)
      {
        from0 = j + flip_block_size * ( (n_threads - (tID + 1) ) * 2 );
        from1 = j + flip_block_size * ( (n_threads - (tID + 1) ) * 2 + 1 );

        to0   = j + flip_block_size * ( (tID + 1) * 2 - 2 );
        to1   = j + flip_block_size * ( (tID + 1) * 2 - 2 + 1 );

        // Direct pivoting
        pivot[to0] = from0;
        pivot[to1] = from1;
      }
    }
  }

  return(0);
}

int OptimizeMatrix(SparseMatrix & A, local_int_t * &pivot, const int& n_max_threads)
{

  // IMPORTANT NOTE:
  // A.mtxIndG is never used in the run, so we do not need to adapt it here.
  // Same for A.globalToLocalMap
  // Indeed the idea is that we optimize (reorder) the local problem and we do not care about the global one.

  const int nrows = A.localNumberOfRows;

  // Variable to permute matrix
  char         *  new_nonzerosInRow  = new char[nrows];
  double       ** new_matrixDiagonal = new double*[nrows];
  double       ** new_matrixValues   = new double*[nrows];
//  global_int_t ** new_mtxIndG        = new global_int_t*[nrows];
  local_int_t  ** new_mtxIndL        = new local_int_t* [nrows];
  std::vector< global_int_t > new_localToGlobalMap(nrows);

  // TODO CHECK IF POSSIBLE TO AVOID THIS SERIALIZATION
#if defined(__HAVE_NEW_DEFAULT_MEM_ALLOC)
  const local_int_t numberOfNonzerosPerRow = 27;

  new_matrixValues[0] = new double[numberOfNonzerosPerRow*nrows];
  new_mtxIndL[0] = new local_int_t[numberOfNonzerosPerRow*nrows];
  //new_mtxIndG[0] = new global_int_t[numberOfNonzerosPerRow*nrows];
  for (local_int_t i = 0 ; i < nrows ; ++i)
  {
      local_int_t p = pivot[i];

      new_nonzerosInRow[i] = A.nonzerosInRow[p];

      new_matrixValues[i] =  new_matrixValues[0]   + i*numberOfNonzerosPerRow;
      new_mtxIndL[i] = new_mtxIndL[0] + i*numberOfNonzerosPerRow;
  //    new_mtxIndG[i] = new_mtxIndG[0] + i*numberOfNonzerosPerRow;
  }
#else
  for (local_int_t i = 0 ; i < nrows ; ++i)
  {
    local_int_t p = pivot[i];

    new_nonzerosInRow[i] = A.nonzerosInRow[p];

    new_matrixValues[i] = new double      [new_nonzerosInRow[i]];
    new_mtxIndL[i]      = new local_int_t [new_nonzerosInRow[i]];
//    new_mtxIndG[i]      = new global_int_t[new_nonzerosInRow[i]];

    delete [] A.matrixValues[i];
//    delete [] A.mtxIndG[i];
  }
#endif

  // Swap and delete
  std::swap( A.nonzerosInRow, new_nonzerosInRow);
  delete [] new_nonzerosInRow;

  std::swap( A.matrixValues, new_matrixValues);
  delete [] new_matrixValues;

//  std::swap( A.mtxIndG, new_mtxIndG);
//  delete [] new_mtxIndG;

  // Start 2nd parallel region
  #pragma omp parallel
  {
    int tID = omp_get_thread_num();

    // Permute matrix (keep the same storage structure - just pivoting)
    local_int_t local_ID;
    std::vector<local_int_t> sortedColumns;
    for (local_int_t i = A.optimizationData[tID][0] ; i <= A.optimizationData[tID][1] ; ++i)
    {
      local_int_t p = pivot[i];

      new_localToGlobalMap[i] = A.localToGlobalMap[p];

      // Sort columns after pivoting
      sortedColumns.clear();
      sortedColumns.reserve(27);
      for (local_int_t j = 0 ; j < A.nonzerosInRow[i] ; ++j)
      {
       local_ID = A.mtxIndL[p][j];
        if (local_ID >= nrows)
          sortedColumns.push_back( local_ID ); // DO NOT PIVOT EXTERNAL ELEMENTS!!!
        else
          sortedColumns.push_back( pivot[local_ID] );
      }
      std::sort(sortedColumns.begin(), sortedColumns.end(), pairComparator);

      for (local_int_t j = 0 ; j < A.nonzerosInRow[i] ; ++j)
      {
        new_mtxIndL[i][j]    = sortedColumns[j];
        A.matrixValues[i][j] = -1;

        // Diagonal
        if (new_mtxIndL[i][j] == i)
        {
          A.matrixValues[i][j]  = 26;
          new_matrixDiagonal[i] = &(A.matrixValues[i][j]);
        }
      }
#if !defined (__HAVE_ELLPACK_FORMAT)
      // Set to zero all other elements
      for (local_int_t j = A.nonzerosInRow[i] ; j < MAX_ELEMENTS_PER_ROW ; ++j)
      {
        new_mtxIndL[i][j]    = 0;
        A.matrixValues[i][j] = 0.;
      }
#endif
      // Find Forward Sweep extra-diag dependency
      if (A.optimizationData[tID][0] > new_mtxIndL[i][0])
      {
        A.optimizationData[tID][6] = std::min(A.optimizationData[tID][6],i);

        if (tID > 0)
        {
          // Search for rightmost extra-diagonal element
          local_ID = 0;
          for (local_int_t j = local_ID + 1; j < A.nonzerosInRow[i]; ++j)
          {
            if (A.optimizationData[tID][0] > new_mtxIndL[i][j])
              local_ID  = j;
            else
              break;
          }
          local_ID = new_mtxIndL[i][local_ID] + 1;
          A.optimizationData[tID-A.optimizationDataDelta[tID]][4] = std::max(A.optimizationData[tID-A.optimizationDataDelta[tID]][4],local_ID);
        }
      }

      // Find Backward Sweep extra-diag dependency
      // Element outside local domain must be skipped here
      local_int_t right_id = A.nonzerosInRow[i]-1;
      for ( ; right_id > -1 ; --right_id)
        if (new_mtxIndL[i][right_id] < nrows )
          break;

      if (A.optimizationData[tID][1] < new_mtxIndL[i][right_id])
      {
        A.optimizationData[tID][12] = std::max(A.optimizationData[tID][12],i);

        if (tID < (n_max_threads - A.optimizationDataDelta[tID]))
        {
          // Search for leftmost extra-diagonal element
          local_int_t local_ID = right_id;
          for (local_int_t j = local_ID - 1; j > -1; --j)
          {
            if (A.optimizationData[tID][1] < new_mtxIndL[i][j])
              local_ID  = j;
            else
              break;
          }
          local_ID = new_mtxIndL[i][local_ID] - 1;
          A.optimizationData[tID+A.optimizationDataDelta[tID]][10] = std::min(A.optimizationData[tID+A.optimizationDataDelta[tID]][10],local_ID);
        }
      }
    }

    for (local_int_t i = A.optimizationData[tID][0] ; i <= A.optimizationData[tID][1] ; ++i)
    {
      // Copy the new LocalToGlobalMap
      A.localToGlobalMap[i] = new_localToGlobalMap[i];

/*
      // Fix GlobalToLocalMap using new LocalToGlobalMap
      A.globalToLocalMap[new_localToGlobalMap[i]] = i;

      // Fix mtxIndG using new LocalToGlobalMap
      for (local_int_t j = 0 ; j < A.nonzerosInRow[i] ; ++j)
      {
        local_ID = new_mtxIndL[i][j];
        if (local_ID >= nrows)
          // TODO This needs to be implemented
        else
          A.mtxIndG[i][j] = new_localToGlobalMap[local_ID];
      }
*/
    }
  }

#if !defined(__HAVE_NEW_DEFAULT_MEM_ALLOC)
  // Delete Local Index
  for (local_int_t i = 0 ; i < nrows ; ++i)
  {
    delete [] A.mtxIndL[i];
  }
#endif

  // Swap and delete
  std::swap( A.matrixDiagonal, new_matrixDiagonal);
  delete [] new_matrixDiagonal;

  std::swap( A.mtxIndL, new_mtxIndL);
  delete [] new_mtxIndL;

  // Pivot ID of elements to send during ExchangeHalo
  for (local_int_t j = 0 ; j < A.totalToBeSent ; ++j)
    A.elementsToSend[j] = pivot[A.elementsToSend[j]];

  return(0);
}


int OptimizeVectors(SparseMatrix& A, Vector& b,Vector& x,Vector& xexact, local_int_t * &pivot, const int& n_max_threads)
{
  // Permute arrays
//  Vector new_b;
//  Vector new_x;
//  Vector new_xexact;

  // Initialize new arrays
//  InitializeVector(new_b,      b.localLength);
//  InitializeVector(new_x,      x.localLength);
//  InitializeVector(new_xexact, xexact.localLength);

  #pragma omp parallel for schedule(static)
  for (local_int_t i = 0 ; i < b.localLength ; ++i)
  {
    b.values[i] = 26.0 - ((double) (A.nonzerosInRow[i]-1));
//    local_int_t p = pivot[i];

//    new_b.values[i]      = b.values[p];
//    new_x.values[i]      = x.values[p];
//    new_xexact.values[i] = xexact.values[p];
  }

//  CopyVector(new_b,b);
//  CopyVector(new_x,x);
//  CopyVector(new_xexact,xexact);

//  DeleteVector(new_b);
//  DeleteVector(new_x);
//  DeleteVector(new_xexact);

  return(0);
}

int InitializeVectorsOptimizationData(Vector& b, Vector& x, Vector& xexact, const int& nrows, const int& n_threads, const int& n_max_threads)
{
  int block_size = nrows / n_threads;

  // Arrays threads optimization structure
  // This must be available to all threads: out of parallel region
  // We need the first 2 values also in case only 1 thread is available
  b.optimizationData       = new local_int_t*[n_max_threads];
  x.optimizationData       = new local_int_t*[n_max_threads];
  xexact.optimizationData  = new local_int_t*[n_max_threads];
  allocatedMemory2Optimize += ( 3 * n_max_threads * 2 ) * sizeof (local_int_t);

  for (int i = 0, j = 0; i < n_max_threads; ++i)
  {
    // Allocate and initialize to zero
    b.optimizationData[i]       = new local_int_t[2]();
    x.optimizationData[i]       = new local_int_t[2]();
    xexact.optimizationData[i]  = new local_int_t[2]();

    if ( i % (n_max_threads / n_threads) == 0 )
    {
      // [0]
      b.optimizationData[i][0]       = block_size * j;
      x.optimizationData[i][0]       = block_size * j;
      xexact.optimizationData[i][0]  = block_size * j;

      // [1]
      b.optimizationData[i][1]       = block_size * (j + 1) - 1;
      x.optimizationData[i][1]       = block_size * (j + 1) - 1;
      xexact.optimizationData[i][1]  = block_size * (j + 1) - 1;

      ++j;
    }
    else
    {
      // [0]
      b.optimizationData[i][0]       = 0;
      x.optimizationData[i][0]       = 0;
      xexact.optimizationData[i][0]  = 0;

      // [1]
      b.optimizationData[i][1]       = -1;
      x.optimizationData[i][1]       = -1;
      xexact.optimizationData[i][1]  = -1;
    }

  }

  return(0);
}

int InitializeCGVectorsOptimizationData(CGData & data, const int& nrows, const int& n_max_threads)
{
  int block_size = nrows / n_max_threads;

  // Arrays threads optimization structure
  // This must be available to all threads: out of parallel region
  // We need the first 2 values also in case only 1 thread is available
  data.r.optimizationData  = new local_int_t*[n_max_threads];
  data.z.optimizationData  = new local_int_t*[n_max_threads];
  data.p.optimizationData  = new local_int_t*[n_max_threads];
  data.Ap.optimizationData = new local_int_t*[n_max_threads];
  allocatedMemory2Optimize += ( 4 * n_max_threads * 2 ) * sizeof (local_int_t);

  for (int i = 0; i < n_max_threads; ++i)
  {
    // Allocate and initialize to zero
    data.r.optimizationData[i]  = new local_int_t[2]();
    data.z.optimizationData[i]  = new local_int_t[2]();
    data.p.optimizationData[i]  = new local_int_t[2]();
    data.Ap.optimizationData[i] = new local_int_t[2]();

    // [0]
    data.r.optimizationData[i][0]  = block_size * i;
    data.z.optimizationData[i][0]  = block_size * i;
    data.p.optimizationData[i][0]  = block_size * i;
    data.Ap.optimizationData[i][0] = block_size * i;

    // [1]
    data.r.optimizationData[i][1]  = block_size * (i + 1) - 1;
    data.z.optimizationData[i][1]  = block_size * (i + 1) - 1;
    data.p.optimizationData[i][1]  = block_size * (i + 1) - 1;
    data.Ap.optimizationData[i][1] = block_size * (i + 1) - 1;
  }

  return(0);
}

void OptimizeMatrixSortRows(SparseMatrix & A)
{
  #pragma omp parallel
  {
    int tID = omp_get_thread_num();

    std::vector<local_int_t> sortedColumns;
    for (local_int_t i = A.optimizationData[tID][0] ; i <= A.optimizationData[tID][1] ; ++i)
    {
      sortedColumns.clear();
      sortedColumns.reserve(27);
      for (local_int_t j = 0 ; j < A.nonzerosInRow[i] ; ++j)
        sortedColumns.push_back( A.mtxIndL[i][j] );
      std::sort(sortedColumns.begin(), sortedColumns.end(), pairComparator);

      for (local_int_t j = 0 ; j < A.nonzerosInRow[i] ; ++j)
      {
        A.mtxIndL[i][j] = sortedColumns[j];
        A.matrixValues[i][j] = -1;

        // Diagonal
        if (sortedColumns[j] == i)
        {
          A.matrixValues[i][j] = 26;
          A.matrixDiagonal[i]  = &(A.matrixValues[i][j]);
        }
      }
    }
  }
}

#if defined (__HAVE_ELLPACK_FORMAT)
void OptimizeMatrixFormatEllpack(SparseMatrix & A)
{
  const int nrows = A.localNumberOfRows;

  // Allocate
//  A.optimizedEllpackDiag = new double*[nrows];
  A.optimizedEllpackVals = new double[ELLPACK_SIZE*nrows];
  A.optimizedEllpackCols = new local_int_t[ELLPACK_SIZE*nrows];
  allocatedMemory2Optimize += ( ELLPACK_SIZE*nrows )        * sizeof (local_int_t) +
//                              ( ELLPACK_SIZE*nrows + nrows) * sizeof (double);
                              ( ELLPACK_SIZE*nrows ) * sizeof (double);

  #pragma omp parallel
  {
    int tID = omp_get_thread_num();
    for (local_int_t i = A.optimizationData[tID][0] ; i <= A.optimizationData[tID][1] ; ++i)
    {
      local_int_t j = 0;
#if 0
      for (; j < A.nonzerosInRow[i] ; ++j)
      {
        A.optimizedEllpackVals[i*ELLPACK_SIZE+j] = A.matrixValues[i][j];
        A.optimizedEllpackCols[i*ELLPACK_SIZE+j] = A.mtxIndL[i][j];

        if (A.mtxIndL[i][j] == i)
          A.optimizedEllpackDiag[i] = &(A.optimizedEllpackVals[i*ELLPACK_SIZE+j]);
      }
      for (; j < ELLPACK_SIZE ; ++j)
      {
        A.optimizedEllpackVals[i*ELLPACK_SIZE+j] = 0.;
        A.optimizedEllpackCols[i*ELLPACK_SIZE+j] = 0;
      }
#else
      for (; j < A.nonzerosInRow[i] ; ++j)
      {
        if (A.mtxIndL[i][j] < i)
        {
          A.optimizedEllpackCols[i*ELLPACK_SIZE+j+1] = A.mtxIndL[i][j];
          A.optimizedEllpackVals[i*ELLPACK_SIZE+j+1] = A.matrixValues[i][j];
        }
        if(A.mtxIndL[i][j] == i)
        {
          A.optimizedEllpackCols[i*ELLPACK_SIZE] = A.mtxIndL[i][j];
          A.optimizedEllpackVals[i*ELLPACK_SIZE] = A.matrixValues[i][j];
          //A.optimizedEllpackDiag[i] = &(A.optimizedEllpackVals[i*ELLPACK_SIZE]);
        }
        if (A.mtxIndL[i][j] > i)
        {
          A.optimizedEllpackCols[i*ELLPACK_SIZE+j] = A.mtxIndL[i][j];
          A.optimizedEllpackVals[i*ELLPACK_SIZE+j] = A.matrixValues[i][j];
        }
      }
      for (; j < ELLPACK_SIZE-1 ; ++j)
      {
        A.optimizedEllpackVals[i*ELLPACK_SIZE+j] = 0.;
        A.optimizedEllpackCols[i*ELLPACK_SIZE+j] = 0;
      }
#if !defined(__HAVE_ELLPACK_FORMAT_WITHOUTDIAG)
      // Adding the inverse of the diagonal as last column
//      A.optimizedEllpackVals[i*ELLPACK_SIZE+j] = 1. / (*(A.optimizedEllpackDiag[i]));
      A.optimizedEllpackVals[i*ELLPACK_SIZE+j] = 1. / A.optimizedEllpackVals[i*ELLPACK_SIZE];
#endif
#endif
    }
  }

  // These data are now useless
  // TODO - in principle we can store optimizedEllpackVals in matrixValues and optimizedEllpackCols in mtxIndL, but does it make sense?
  if (A.mtxIndL) {
      delete [] A.mtxIndL;
      A.mtxIndL = 0;
  }
  if (A.matrixValues) {
      delete [] A.matrixValues;
      A.matrixValues = 0;
  }
  if (A.matrixDiagonal) {
      delete [] A.matrixDiagonal;
      A.matrixDiagonal = 0;
  }

  allocatedMemory2Optimize -= ( MAX_ELEMENTS_PER_ROW*nrows ) * sizeof (local_int_t) +
//                              ( ELLPACK_SIZE*nrows + nrows) * sizeof (double);
                              ( MAX_ELEMENTS_PER_ROW*nrows + nrows ) * sizeof (double);
}
#endif

/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact)
{
  // Initialize the allocated memory
  allocatedMemory2Optimize = 0;

#ifndef HPCG_NO_OPENMP

  // Check if we have multiple threads
  int n_max_threads = omp_get_max_threads();

  // Time container
  double time[7][2];

  // Initialize Optimization Data (need also with 1 single thread)
  time[0][0] = mytimer();
  InitializeMatrixOptimizationData(A, n_max_threads, n_max_threads);
  time[0][1] = mytimer();

  if (n_max_threads > 1)
  {
    // Initialize pivot
    time[1][0] = mytimer();
    local_int_t * pivot;
    ComputePivot(pivot, A.localNumberOfRows, n_max_threads);
    time[1][1] = mytimer();

    // Optimize Matrix
    time[2][0] = mytimer();
    OptimizeMatrix(A, pivot, n_max_threads);
    time[2][1] = mytimer();

    // Optimize Vectors
    time[3][0] = mytimer();
    OptimizeVectors(A, b, x, xexact, pivot, n_max_threads);
    time[3][1] = mytimer();

    // Delete pivoting
    delete[] pivot;
  }
  else
  {
    // For the Ellpack we need at least to sort the rows
    OptimizeMatrixSortRows(A);
  }

  // Sub level
  time[5][0] = mytimer();

  // Scale factor for sublevels
  int factor = 2;
  int levelThreads = 1;
  if (n_max_threads > 1)
    levelThreads = n_max_threads / factor; // We use always half-threads with respect to the level above
  MGData * levelData = A.mgData;
  SparseMatrix * levelMatrix = A.Ac;

  while (levelMatrix != 0)
  {
    if (levelThreads > 1)
    {
      // Initialize pivot
      local_int_t * subpivot;
      ComputePivot(subpivot, levelMatrix->localNumberOfRows, levelThreads);

      // Initialize Optimization Data (need also with 1 single thread)
      InitializeMatrixOptimizationData(*levelMatrix, levelThreads, n_max_threads);

      // Optimize Matrix
      OptimizeMatrix(*levelMatrix, subpivot, n_max_threads);

      // Finalize Optimization Data (need also with 1 single thread)
      FinalizeMatrixOptimizationData(*levelMatrix, n_max_threads);

      // Delete pivoting
      delete[] subpivot;

      // Initialize Optimization Data (need also with 1 single thread)
      InitializeVectorsOptimizationData(*levelData->rc, *levelData->xc, *levelData->Axf, levelMatrix->localNumberOfRows, levelThreads, n_max_threads);

      // Next level
      levelThreads /= factor;
      levelData = levelMatrix->mgData;
    }
    else
    {
      // Initialize Optimization Data (need also with 1 single thread)
      InitializeMatrixOptimizationData(*levelMatrix, levelThreads, n_max_threads);

      OptimizeMatrixSortRows(*levelMatrix);
    }

    // Next level
    levelMatrix = levelMatrix->Ac;
  }

  time[5][1] = mytimer();

  // Finalize Optimization Data
  time[0][0] += mytimer();
  if (n_max_threads > 1)
    FinalizeMatrixOptimizationData(A, n_max_threads);
  time[0][1] += mytimer();

  // Initialize Optimization Data for Vectors(need also with 1 single thread)
  time[4][0] = mytimer();
  InitializeVectorsOptimizationData(b, x, xexact, A.localNumberOfRows, n_max_threads, n_max_threads);
  InitializeCGVectorsOptimizationData(data, A.localNumberOfRows, n_max_threads);
  time[4][1] = mytimer();

  // Ellpack format
#if defined (__HAVE_ELLPACK_FORMAT)
  time[6][0] = mytimer();
  OptimizeMatrixFormatEllpack(A);
  levelMatrix = A.Ac;
  while (levelMatrix != 0)
  {
    OptimizeMatrixFormatEllpack(*levelMatrix);
    levelMatrix = levelMatrix->Ac;
  }
  time[6][1] = mytimer();
#endif

  // DEBUG
#if 1 //defined(DEBUG_PIVOTING)

  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  if (rank == 0)
  {
      printf("Init./Fin. matrix OperationData done in:  %22.16e s\n",time[0][1]-time[0][0]);
      printf("Compute pivot done in:                    %22.16e s\n",time[1][1]-time[1][0]);
      printf("Optimize matrix done in:                  %22.16e s\n",time[2][1]-time[2][0]);
      printf("Optimize vectors done in:                 %22.16e s\n",time[3][1]-time[3][0]);
      printf("Initialize vectors OperationData done in: %22.16e s\n",time[4][1]-time[4][0]);
      printf("Sublevels (all together) in:              %22.16e s\n",time[5][1]-time[5][0]);
#if defined (__HAVE_ELLPACK_FORMAT)
      printf("Ellpack (all together) in:                %22.16e s\n\n",time[6][1]-time[6][0]);
#else
      printf("\n");
#endif

      printf("Matrix: \n");
      for (int i = 0; i < n_max_threads; ++i)
      {
        if ((A.optimizationData[i][1] - A.optimizationData[i][0] + 1) % 4)
          printf("thread: %2d, block: [%5d %5d], 1st FW: [%5d %5d], 2nd FW: [%5d %5d], 3rd FW: [%5d %5d],\n"
                 "                                  1st BW: [%5d %5d], 2nd BW: [%5d %5d], 3rd BW: [%5d %5d] - WARNING RANGE NOT DIVISIBLE BY 4\n",i,
                                                                                       A.optimizationData[i][0],
                                                                                       A.optimizationData[i][1],
                                                                                       A.optimizationData[i][2],
                                                                                       A.optimizationData[i][3],
                                                                                       A.optimizationData[i][4],
                                                                                       A.optimizationData[i][5],
                                                                                       A.optimizationData[i][6],
                                                                                       A.optimizationData[i][7],
                                                                                       A.optimizationData[i][8],
                                                                                       A.optimizationData[i][9],
                                                                                       A.optimizationData[i][10],
                                                                                       A.optimizationData[i][11],
                                                                                       A.optimizationData[i][12],
                                                                                       A.optimizationData[i][13]);
        else
          printf("thread: %2d, block: [%5d %5d], 1st FW: [%5d %5d], 2nd FW: [%5d %5d], 3rd FW: [%5d %5d],\n"
                 "                                  1st BW: [%5d %5d], 2nd BW: [%5d %5d], 3rd BW: [%5d %5d]\n",i,
                                                                                       A.optimizationData[i][0],
                                                                                       A.optimizationData[i][1],
                                                                                       A.optimizationData[i][2],
                                                                                       A.optimizationData[i][3],
                                                                                       A.optimizationData[i][4],
                                                                                       A.optimizationData[i][5],
                                                                                       A.optimizationData[i][6],
                                                                                       A.optimizationData[i][7],
                                                                                       A.optimizationData[i][8],
                                                                                       A.optimizationData[i][9],
                                                                                       A.optimizationData[i][10],
                                                                                       A.optimizationData[i][11],
                                                                                       A.optimizationData[i][12],
                                                                                       A.optimizationData[i][13]);

      }

      int levelThreads = n_max_threads / factor; // We use always half-threads with respect to the level above
      SparseMatrix * levelMatrix = A.Ac;
      while (levelMatrix != 0)
      {
          printf("Sub-level: \n");
          for (int i = 0; i < n_max_threads; ++i)
          {
            if ((levelMatrix->optimizationData[i][1] - levelMatrix->optimizationData[i][0] + 1) % 4)
              printf("thread: %2d, block: [%5d %5d], 1st FW: [%5d %5d], 2nd FW: [%5d %5d], 3rd FW: [%5d %5d],\n"
                       "                                  1st BW: [%5d %5d], 2nd BW: [%5d %5d], 3rd BW: [%5d %5d] - WARNING RANGE NOT DIVISIBLE BY 4\n",i,
                                                                                             levelMatrix->optimizationData[i][0],
                                                                                             levelMatrix->optimizationData[i][1],
                                                                                             levelMatrix->optimizationData[i][2],
                                                                                             levelMatrix->optimizationData[i][3],
                                                                                             levelMatrix->optimizationData[i][4],
                                                                                             levelMatrix->optimizationData[i][5],
                                                                                             levelMatrix->optimizationData[i][6],
                                                                                             levelMatrix->optimizationData[i][7],
                                                                                             levelMatrix->optimizationData[i][8],
                                                                                             levelMatrix->optimizationData[i][9],
                                                                                             levelMatrix->optimizationData[i][10],
                                                                                             levelMatrix->optimizationData[i][11],
                                                                                             levelMatrix->optimizationData[i][12],
                                                                                             levelMatrix->optimizationData[i][13]);
            else
              printf("thread: %2d, block: [%5d %5d], 1st FW: [%5d %5d], 2nd FW: [%5d %5d], 3rd FW: [%5d %5d],\n"
                     "                                  1st BW: [%5d %5d], 2nd BW: [%5d %5d], 3rd BW: [%5d %5d]\n",i,
                                                                                           levelMatrix->optimizationData[i][0],
                                                                                           levelMatrix->optimizationData[i][1],
                                                                                           levelMatrix->optimizationData[i][2],
                                                                                           levelMatrix->optimizationData[i][3],
                                                                                           levelMatrix->optimizationData[i][4],
                                                                                           levelMatrix->optimizationData[i][5],
                                                                                           levelMatrix->optimizationData[i][6],
                                                                                           levelMatrix->optimizationData[i][7],
                                                                                           levelMatrix->optimizationData[i][8],
                                                                                           levelMatrix->optimizationData[i][9],
                                                                                           levelMatrix->optimizationData[i][10],
                                                                                           levelMatrix->optimizationData[i][11],
                                                                                           levelMatrix->optimizationData[i][12],
                                                                                           levelMatrix->optimizationData[i][13]);
          }

          levelThreads /= factor;
          levelMatrix = levelMatrix->Ac;
      }


      printf("\nVectors: \n");
      for (int i = 0; i < n_max_threads; ++i)
      {
        printf("thread: %d, block: [%d %d]\n",i,b.optimizationData[i][0],
                                                b.optimizationData[i][1]);

#if defined(__IBMC__) || defined(__IBMCPP__)
        if ((b.optimizationData[i][1]-b.optimizationData[i][0]+1) % 4 != 0)
          printf("WARNING - SIMD ISSUE : block [%d %d] is not divisible by 4\n",b.optimizationData[i][0],b.optimizationData[i][1]);
#endif

      }
    printf("\n");
  }
#endif

#endif

  return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return static_cast<double> (allocatedMemory2Optimize * size);

}
