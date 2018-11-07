
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
 @file Vector.hpp

 HPCG data structures for dense vectors
 */

#ifndef VECTOR_HPP
#define VECTOR_HPP
#include <cassert>
#include <cstdlib>
#include "Geometry.hpp"

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

struct Vector_STRUCT {
  local_int_t localLength;  //!< length of local portion of the vector
  double * values;          //!< array of values
  /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  local_int_t ** optimizationData;

};
typedef struct Vector_STRUCT Vector;

/*!
  Initializes input vector.

  @param[in] v
  @param[in] localLength Length of local portion of input vector
 */
inline void InitializeVector(Vector & v, local_int_t localLength) {
  v.localLength = localLength;
  v.values = new double[localLength];
  v.optimizationData = 0;
  return;
}

/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
inline void ZeroVector(Vector & v) {
  local_int_t localLength = v.localLength;
  double * vv = v.values;
  for (int i=0; i<localLength; ++i) vv[i] = 0.0;
  return;
}
/*!
  Multiply (scale) a specific vector entry by a given value.

  @param[inout] v Vector to be modified
  @param[in] index Local index of entry to scale
  @param[in] value Value to scale by
 */
inline void ScaleVectorValue(Vector & v, local_int_t index, double value) {
  assert(index>=0 && index < v.localLength);
  double * vv = v.values;
  vv[index] *= value;
  return;
}
/*!
  Fill the input vector with pseudo-random values.

  @param[in] v
 */
inline void FillRandomVector(Vector & v) {
  local_int_t localLength = v.localLength;
  double * vv = v.values;
  for (int i=0; i<localLength; ++i) vv[i] = rand() / (double)(RAND_MAX) + 1.0;
  return;
}
/*!
  Copy optimize data

  @param[in] v Input vector
  @param[in] w Output vector
 */
inline void CopyOptimizations(const Vector & v, Vector & w) {
  local_int_t localLength = v.localLength;
  assert(w.localLength >= localLength);

#ifndef HPCG_NO_OPENMP
  if (v.optimizationData)
  {
    int n_threads = omp_get_max_threads();
    if (!w.optimizationData)
    {
        w.optimizationData = new local_int_t*[n_threads];
        for (int i = 0; i < n_threads; ++i)
          w.optimizationData[i] = new local_int_t[2]();
    }
    for (int i = 0; i < n_threads; ++i)
    {
      w.optimizationData[i][0] = v.optimizationData[i][0];
      w.optimizationData[i][1] = v.optimizationData[i][1];
    }
  }
#endif

  return;
}
/*!
  Copy input vector to output vector.

  @param[in] v Input vector
  @param[in] w Output vector
 */
inline void CopyVector(const Vector & v, Vector & w) {
  local_int_t localLength = v.localLength;
  assert(w.localLength >= localLength);
  double * vv = v.values;
  double * wv = w.values;
  for (int i=0; i<localLength; ++i) wv[i] = vv[i];

#ifndef HPCG_NO_OPENMP
  CopyOptimizations(v, w);
#endif

  return;
}
/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteVector(Vector & v) {

  delete [] v.values;
  v.localLength = 0;

#ifndef HPCG_NO_OPENMP
  // Delete optimization data
  if (v.optimizationData)
  {
    for (int i = 0; i < omp_get_max_threads(); ++i)
      delete [] v.optimizationData[i];
    delete [] v.optimizationData;
  }
#endif

  return;
}

#endif // VECTOR_HPP
