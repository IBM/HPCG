################################################################################
# Optimized High Performance Conjugate Gradient Benchmark (HPCG) for IBM BG/Q and IBM POWER9
################################################################################


## 1. Introduction ##
HPCG is a software package that performs a fixed number of multigrid preconditioned
(using a symmetric Gauss-Seidel smoother) conjugate gradient (PCG) iterations.

We present an optimized CPU-only version of HPCG on IBM processors. It features the following optimizations:
- explicit SIMD vectorization, data prefetching, asynchronous MPI communication
- smart pivoting: new OpenMP parallelization approach for SYMGS, the most time consuming kernel of HPCG

Fine tuning of several parameters, such as
- Local problem size
- Number of MPI tasks and OpenMP threads

The code achieves 1.6% of the peak performance on BG/Q and more than 3% on POWER9.

## 2. Optimizations ##
### 2.1 IBM BG/Q
#### a. Smart pivoting for parallel SYSMGS
A parallel implementation of SYMGS requires coloring, which has two undesired side effects: a) slows down convergence, b) limit cache reuse.

Proposed solution: stencil discretization leads to a uniform (diagonal) matrix structure: we can rely on that
observation to enable multithreaded parallelization. The idea of this approach is illustrated in the following image:

![](./SmartPivoting.png)

#### b. Fine tuning
- Strong compiler optimization (-O5, -qipa=level=2, -qhot=level=2, etc) but with little effect overall
- AXPY and DOT manually SIMD vectorized (slightly better performances than auto-compiled versions)
- Contiguous storage for matrix (to help hardware prefetching – very important on BG/Q)
- Improve backward prefetching of Gauss-Seidel smoother with `__dcbt` instructions
- Manual unrolling factor of 2 for SpMV, and slightly improved code
- Use `Isend/Irecv` with a single `MPI_Waitall` call at the end (better overlap of communications)
- Optimized local problem size
  - A smaller problem size gives better MG and SPMV performance (better cache use)
    ... however it also increases DOT `MPI_Allreduce` (due to wait time) – need to find the best compromise!
  - We use `48x16x16` or `56x16x16` (on few racks) and `96x32x32` or `112x32x32` (on many racks)

### 2.2 IBM POWER9
- The XLC (16.1.0) compiler optimizations for BG/Q remain the same  - still little effect overall
- MPI configuration:
  - One task per core
  - Binding policy: `mpirun --bind–to core --map-by core`
- OpenMP configuration:
  - two threads per task (core)
  - `OMP_PROC_BIND=FALSE` (no explicit binding to the hardware threads of the core)
  - `OMP_WAIT_POLICY=ACTIVE` (no need for yield)
- Problem size
  - Local domain is set to `160x160x96` (or `160x96x160`)
  - Larger than BG/Q due to higher memory bandwidth

## 3. Results
### 3.1 Vulcan @ LLNL (BG/Q)
- https://www.top500.org/system/177732
- 24K nodes
- 32 MPI tasks/node, 2 threads each
- Local domain dimension: 112x32x32
- 80.9 TFlop/s (3.29 GFlop/s per node)
- Fraction of peak performance: 1.6%

### 3.2 Sequoia @ LLNL (BG/Q)
- https://www.top500.org/system/177556
- 96K nodes
- 32 MPI tasks/node, 2 threads each
- Local domain dimension: 112x32x32
- 330.4 Tflop/s (3.36 GFlop/s per node)
  - #10 in HPGC results list (June 2018)
- Fraction of peak performance: 1.6%


### 3.3 Marenostrum P9 CTE @ BSC (Power9)
#### System
- https://www.top500.org/system/179442
  - #255 in TOP500 (June 2018)
- 54 nodes organized in 3 racks
- 52 compute and 2 login nodes
- Each node:
 - 2x IBM POWER9 20C 3.1GHz
 - 40 cores with four-way multithreading variant SMT4
 - 4x NVIDIA Tesla V100
- Interconnection network: Dual-rail Mellanox EDR Infiniband

### Experiments
- 32 nodes
- 40 MPI tasks/node, 2 threads each
- Local domain dimension: 160x160x96
- 1193.4 GFlop/s (37.3 GFlop/s per node)
- Fraction of peak: > 3% (depends on the actual core frequency)

![](./BSC-P9-CTE-scaling.png)

## 4. Installation and Testing ##
You can read the general instructions provided in the file `INSTALL` in this directory.

According to them, you can create a custom directory, ``build`` in
this example, for the results of compilation and linking:

    mkdir build

Next, go this new directory and use the ``configure`` script to create the
build infrastructure:

    cd build
    ./configure <arch>

For IBM platforms, `<arch>` can be `bgq`, `p8` or `p9`, for BG/Q, Power8 and Power9 respectively.

You do not have to modify any of the compile time options that have been already set.
The optimized HPCG version is compiled with MPI and OpenMP enabled.


## 5. BOF presentation at SC18
The performance results of the optimized HPCG version on IBM Power9 were presented at a [BOF session at SC18]( https://www.hpcg-benchmark.org/custom/index.html?lid=154&slid=298).
The presentation slides are available here: [Porting Optimized HPCG 3.1 for IBM BG/Q to IBM POWER9](https://www.hpcg-benchmark.org/downloads/sc18/HPCG_IBM_P9_v05.pdf).
