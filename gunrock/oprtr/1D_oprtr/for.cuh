// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * for.cuh
 *
 * @brief Simple "for" operations
 */

#pragma once

#include <cooperative_groups.h>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/util/cuda_properties.cuh>

namespace gunrock {
namespace oprtr {

#define FOR_BLOCKSIZE 256
#define FOR_GRIDSIZE 512

typedef uint64_t ForIterT;

template <typename OpT>
__global__ void For_Kernel(ForIterT loop_size, OpT op) {
  const ForIterT STRIDE = (ForIterT)blockDim.x * gridDim.x;
  for (ForIterT i = (ForIterT)blockDim.x * blockIdx.x + threadIdx.x;
       i < loop_size; i += STRIDE)
    op(i);
}

__global__
void scanDegrees(int size, int *d_degrees, int *incrDegrees) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < size) {
        //write initial values to shared memory
        __shared__ int prefixSum[1024];
        int modulo = threadIdx.x;
        prefixSum[modulo] = d_degrees[thid];
        //if ( d_degrees[thid] > 4 && thid > 205900)printf("th = %d de = %d\n", thid, d_degrees[thid]);
        __syncthreads();

        //calculate scan on this block
        //go up
        for (int nodeSize = 2; nodeSize <= 1024; nodeSize <<= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    int nextPosition = modulo + (nodeSize >> 1);
                    prefixSum[modulo] += prefixSum[nextPosition];
                    //if (modulo == 0) printf("mo = %d\n",prefixSum[0]);
                }
            }
            __syncthreads();
        }

        //write information for increment prefix sums
        if (modulo == 0) {
            int block = thid >> 10;
            incrDegrees[block + 1] = prefixSum[modulo];
            //printf("bl = %d inc = %d\n",block,incrDegrees[block+1]);
        }

        //go down
        for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    int next_position = modulo + (nodeSize >> 1);
                    int tmp = prefixSum[modulo];
                    prefixSum[modulo] -= prefixSum[next_position];
                    prefixSum[next_position] = tmp;

                }
            }
            __syncthreads();
        }
        d_degrees[thid] = prefixSum[modulo];
    }

}

template <typename OpT>
__global__ void For_Kernel1(ForIterT loop_size, OpT op) {

  // calculate no of vertices each thread has to process
  int no_vertices_per_thread = loop_size/(blockDim.x * gridDim.x);
  int thread_no = blockDim.x * blockIdx.x + threadIdx.x;

  // the first vertex this thread operates on, one thread operates on
  // vertex v , v+ 32, v+64 and so on
  int start_v =   ((thread_no/32)*32)*no_vertices_per_thread + min(((thread_no/32)*32),(int)loop_size%(gridDim.x * blockDim.x)) + threadIdx.x%32;
  
  for (int  i = 0; i < no_vertices_per_thread; i++)
    op(i*32 + start_v);

  // Since number of threads may not divide number of vertices exactly,
  // SO if a remainder r is left, the first "r" threads operate on one extra
  // vertex.
  if (thread_no < loop_size%(gridDim.x * blockDim.x)){
    op(no_vertices_per_thread*32+start_v);
  }
}

template <typename OpT>
cudaError_t For(OpT op, ForIterT loop_size, util::Location target,
                cudaStream_t stream = 0,
                int grid_size = util::PreDefinedValues<int>::InvalidValue,
                int block_size = util::PreDefinedValues<int>::InvalidValue) {
  cudaError_t retval = cudaSuccess;

  if ((target & util::HOST) == util::HOST) {
#pragma omp parallel for
    for (ForIterT i = 0; i < loop_size; i++) op(i);
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    if (!util::isValid(grid_size)) grid_size = FOR_GRIDSIZE;
    if (!util::isValid(block_size)) block_size = FOR_BLOCKSIZE;
    // printf("grid_size = %d, block_size = %d\n",
    //    grid_size, block_size);
    if (grid_size == 4 && block_size == 128 )
    {For_Kernel1<<<grid_size, block_size, 0, stream>>>(loop_size, op);}
    else 
    For_Kernel<<<grid_size, block_size, 0, stream>>>(loop_size, op);
  }
  return retval;
}

template <typename OpT>
__global__ void RepeatFor0_Kernel(int num_repeats, ForIterT loop_size, OpT op) {
#if (GR_CUDA_ARCH >= 600)
  const ForIterT STRIDE = (ForIterT)blockDim.x * gridDim.x;
  auto grid = cooperative_groups::this_grid();

  for (int r = 0; r < num_repeats; r++) {
    for (ForIterT i = (ForIterT)blockDim.x * blockIdx.x + threadIdx.x;
         i < loop_size; i += STRIDE)
      op(r, i);
    grid.sync();
  }
#endif
}

template <typename OpT>
cudaError_t RepeatFor0(
    OpT op, int num_repeats, ForIterT loop_size, util::Location target,
    cudaStream_t stream = 0,
    int grid_size = util::PreDefinedValues<int>::InvalidValue,
    int block_size = util::PreDefinedValues<int>::InvalidValue) {
  cudaError_t retval = cudaSuccess;

  if ((target & util::HOST) == util::HOST) {
    for (int r = 0; r < num_repeats; r++) {
#pragma omp parallel for
      for (ForIterT i = 0; i < loop_size; i++) op(r, i);
    }
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    if (!util::isValid(grid_size)) grid_size = FOR_GRIDSIZE;
    if (!util::isValid(block_size)) block_size = FOR_BLOCKSIZE;

    void *kernelArgs[] = {
        (void *)&num_repeats,
        (void *)&loop_size,
        (void *)&op,
    };
    dim3 grid_dim = grid_size;
    dim3 block_dim = block_size;

    // auto func = RepeatFor_Kernel<OpT>;
    retval =
        cudaLaunchCooperativeKernel((void *)RepeatFor0_Kernel<OpT>, grid_dim,
                                    block_dim, kernelArgs, 0, stream);
    if (retval)
      return util::GRError(retval, "RepeatFor kernel launch failed", __FILE__,
                           __LINE__);
  }
  return retval;
}

#if (__CUDACC_VER_MAJOR__ >= 10)
template <typename OpT>
__global__ void RepeatFor1_Kernel(int r, ForIterT loop_size, OpT op) {
  const ForIterT STRIDE = (ForIterT)blockDim.x * gridDim.x;
  {
    for (ForIterT i = (ForIterT)blockDim.x * blockIdx.x + threadIdx.x;
         i < loop_size; i += STRIDE)
      op(r, i);
  }
}

template <typename OpT>
cudaError_t RepeatFor1(
    OpT op, int num_repeats, ForIterT loop_size, util::Location target,
    cudaStream_t stream = 0,
    int grid_size = util::PreDefinedValues<int>::InvalidValue,
    int block_size = util::PreDefinedValues<int>::InvalidValue) {
  cudaError_t retval = cudaSuccess;

  if ((target & util::HOST) == util::HOST) {
    for (int r = 0; r < num_repeats; r++) {
#pragma omp parallel for
      for (ForIterT i = 0; i < loop_size; i++) op(r, i);
    }
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    if (!util::isValid(grid_size)) grid_size = FOR_GRIDSIZE;
    if (!util::isValid(block_size)) block_size = FOR_BLOCKSIZE;
    static cudaGraph_t cuda_graph;
    static cudaGraphExec_t graph_exec;

    static bool first_in = true;

    if (first_in) {
      dim3 grid_dim = grid_size;
      dim3 block_dim = block_size;
      cudaKernelNodeParams kernel_params = {0};
      std::vector<cudaGraphNode_t> node_dependencies;
      if (retval = util::GRError(cudaGraphCreate(&cuda_graph, 0),
                                 "cudaGraphCreate failed", __FILE__, __LINE__))
        return retval;

      for (int r = 0; r < num_repeats; r++) {
        void *kernelArgs[] = {
            (void *)&r,
            (void *)&loop_size,
            (void *)&op,
        };
        cudaGraphNode_t kernel_node;

        kernel_params.func = (void *)RepeatFor1_Kernel<OpT>;
        kernel_params.gridDim = grid_dim;
        kernel_params.blockDim = block_dim;
        kernel_params.sharedMemBytes = 0;
        kernel_params.kernelParams = kernelArgs;
        kernel_params.extra = NULL;

        retval = util::GRError(
            cudaGraphAddKernelNode(&kernel_node, cuda_graph,
                                   node_dependencies.data(),
                                   node_dependencies.size(), &kernel_params),
            "cudaGraphAddKernelNode failed", __FILE__, __LINE__);
        if (retval) return retval;
        node_dependencies.clear();
        node_dependencies.push_back(kernel_node);
      }

      retval = util::GRError(
          cudaGraphInstantiate(&graph_exec, cuda_graph, NULL, NULL, 0),
          "cudaGraphInstantiate failed", __FILE__, __LINE__);
      if (retval) return retval;
      first_in = false;
    }

    retval = util::GRError(cudaGraphLaunch(graph_exec, stream),
                           "cudaGraphLaunch failed", __FILE__, __LINE__);
    if (retval) return retval;
  }
  return retval;
}
#endif

template <typename OpT>
cudaError_t RepeatFor2(
    OpT op, int num_repeats, ForIterT loop_size, util::Location target,
    cudaStream_t stream = 0,
    int grid_size = util::PreDefinedValues<int>::InvalidValue,
    int block_size = util::PreDefinedValues<int>::InvalidValue) {
  cudaError_t retval = cudaSuccess;

  for (int r = 0; r < num_repeats; r++) {
    retval = For([op, r] __host__ __device__(const ForIterT &i) { op(r, i); },
                 loop_size, target, stream, grid_size, block_size);
    if (retval) return retval;
  }
  return retval;
}

template <typename OpT>
cudaError_t RepeatFor(
    OpT op, int num_repeats, ForIterT loop_size, util::Location target,
    cudaStream_t stream = 0,
    int grid_size = util::PreDefinedValues<int>::InvalidValue,
    int block_size = util::PreDefinedValues<int>::InvalidValue,
    int method = 2) {
  cudaError_t retval = cudaSuccess;


  if (method == 0 || target == util::HOST) {
#if (GR_CUDA_ARCH >= 600)
    retval = RepeatFor0(op, num_repeats, loop_size, target, stream, grid_size,
                        block_size);
#else
    util::PrintMsg("Cannot use RepeatFor0 when compiling for SM version < 6.0 - defaulting to RepeatFor2");
    retval = RepeatFor2(op, num_repeats, loop_size, target, stream, grid_size,
	                block_size);
#endif
  }

  else if (method == 1) {
#if (__CUDACC_VER_MAJOR__ >= 10)
    retval = RepeatFor1(op, num_repeats, loop_size, target, stream, grid_size,
                        block_size);
#else
    util::PrintMsg("Cannot use RepeatFor1 when CUDACC < 10 - defaulting to RepeatFor2");
    retval = RepeatFor2(op, num_repeats, loop_size, target, stream, grid_size,
                        block_size);
#endif
  }
  else if (method == 2) {
    retval = RepeatFor2(op, num_repeats, loop_size, target, stream, grid_size,
                        block_size);
  }

  return retval;
}

}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

