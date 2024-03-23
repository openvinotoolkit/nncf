#ifndef _COMMON_CUDA_FUNCS_CUH_
#define _COMMON_CUDA_FUNCS_CUH_

// Have to define common CUDA __device__ funcs in headers because moving them
// to separate translation units will require relocatable device code compilation,
// which is rumoured to degrade performance.

#include "dispatch.h"
#include "common_cuda_defs.cuh"


#define ENABLE_ONLY_FOR_NONREDUCED_FP_TYPES(TYPE_NAME) std::enable_if_t< \
                     std::is_same<float, TYPE_NAME>::value || \
                     std::is_same<double, TYPE_NAME>::value, bool> = true

// Volatile c10::Half and c10::BFloat16 arithmetic is not supported, thus the implicit warp-synchronous
// programming via "volatile" (which is deprecated anyway) cannot be used.
// Using modern explicit intra-warp thread synchronization primitives.
// For more information, see https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/ and
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__shfl#warp-shuffle-functions

template <typename scalar_accum_t, ENABLE_ONLY_FOR_NONREDUCED_FP_TYPES(scalar_accum_t)>
__device__ void sum_warp(scalar_accum_t* sharr) {
    int tidx = threadIdx.x & 31;
    scalar_accum_t v = sharr[tidx];
    v += __shfl_down_sync(-1, v, 16);
    v += __shfl_down_sync(-1, v, 8);
    v += __shfl_down_sync(-1, v, 4);
    v += __shfl_down_sync(-1, v, 2);
    v += __shfl_down_sync(-1, v, 1);
    sharr[tidx] = v;
}


template <typename scalar_accum_t, ENABLE_ONLY_FOR_NONREDUCED_FP_TYPES(scalar_accum_t)>
__device__ inline void gather_warp_execution_results(scalar_accum_t* sharr, const uint16_t tidx) {
    sharr[tidx] = tidx * CUDA_WARP_SIZE < CUDA_MAX_NUM_THREADS_PER_BLOCK ? sharr[tidx * CUDA_WARP_SIZE] : static_cast<scalar_accum_t>(0.0);
}


// Reduces the contents of a shared memory array of CUDA_MAX_NUM_THREADS_PER_BLOCK using
// warp-powered reduction. The final sum will be stored in the 0-th element of the shared memory array.
template <typename scalar_accum_t, ENABLE_ONLY_FOR_NONREDUCED_FP_TYPES(scalar_accum_t)>
__device__ void reduce_in_block_using_warp_sums(scalar_accum_t* __restrict__ sh_mem,
        uint16_t tidx) {
    __syncthreads();
    // Will reduce the summation to CUDA_MAX_WARPS_PER_BLOCK elements that are
    // spaced CUDA_WARP_SIZE elements apart in the shared memory
    sum_warp(sh_mem + (tidx & ~(CUDA_WARP_SIZE - 1)));

    __syncthreads();
    if (tidx < CUDA_MAX_WARPS_PER_BLOCK) {
        // Do warp reduction again - because currently CUDA_MAX_WARPS_PER_BLOCK == CUDA_WARP_SIZE, this
        // will lead to the 0-th element of the shared memory containing the entire per-block sum
        gather_warp_execution_results(sh_mem, tidx);
        sum_warp(sh_mem);
    }
}


__device__ bool last_block(int32_t* counter, uint32_t total_blocks_count) {
    __threadfence();

    int last = 0;
    if (threadIdx.x == 0) {
        last = atomicAdd(counter, 1);
    }

    return __syncthreads_or(last == total_blocks_count - 1);
}


template <typename scalar_t, typename scalar_accum_t = scalar_t, ENABLE_ONLY_FOR_NONREDUCED_FP_TYPES(scalar_accum_t)>
__device__ void reduce_with_shared_memory(
        scalar_accum_t* __restrict__ sh_arr,
        scalar_accum_t current_thread_sum,
        const uint16_t tidx,
        const uint32_t bidx,
        scalar_accum_t* __restrict__ dev_tmp,
        int32_t* __restrict__ dev_last_block_counter,
        scalar_t* __restrict__ grad,
        uint32_t total_number_of_blocks) {

    // Put each thread sum element into shared memory (CUDA_MAX_NUM_THREADS_PER_BLOCK elements in total)
    sh_arr[tidx] = current_thread_sum;

    // Do warp reduction on the entire shared memory of a single block
    reduce_in_block_using_warp_sums(sh_arr, tidx);

    // Store the per-block sum for each block in the pre-allocated array (which has dimensions equal to grid dimensions)
    if (tidx == 0) {
        dev_tmp[bidx] = sh_arr[0];
    }

    // Synchronize blocks and make the last block of the grid do the reduction across the per-block sums
    // to obtain final sums
    if (last_block(dev_last_block_counter, total_number_of_blocks)) {

        // WARNING: seems like this will only work for total number of blocks to reduce across that is < CUDA_MAX_NUM_THREADS_PER_BLOCK
        sh_arr[tidx] = tidx < total_number_of_blocks ? dev_tmp[tidx] : static_cast<scalar_accum_t>(0.0);
        reduce_in_block_using_warp_sums(sh_arr, tidx);

        if (tidx == 0) {
            grad[0] = sh_arr[0];
        }
    }
}


#endif // _COMMON_CUDA_FUNCS_CUH_
