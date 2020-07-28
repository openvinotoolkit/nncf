#ifndef _COMMON_CUDA_FUNCS_CUH_
#define _COMMON_CUDA_FUNCS_CUH_

// Have to define common CUDA __device__ funcs in headers because moving them
// to separate translation units will require relocatable device code compilation,
// which is rumoured to degrade performance.

#include "common_cuda_defs.cuh"

// support only warp size = 32
template <typename scalar_t>
__device__ void sum_warp(volatile scalar_t* sharr) {
    int tidx = threadIdx.x & 31;
    if (tidx < 16) {
        sharr[tidx] += sharr[tidx + 16];
        sharr[tidx] += sharr[tidx + 8];
        sharr[tidx] += sharr[tidx + 4];
        sharr[tidx] += sharr[tidx + 2];
        sharr[tidx] += sharr[tidx + 1];
    }
}

// Since volatile c10::Half arithmetic is not supported, will have to sacrifice
// the implicit warp-synchronous programming in favor of explicit intra-warp thread
// synchronization

template <typename scalar_t>
__device__ void sum_warp_with_explicit_sync(scalar_t* sharr) {
    uint16_t tidx = threadIdx.x & 31;
    if (tidx < 16) {
        sharr[tidx] += sharr[tidx + 16];
    }
    __syncwarp();
    if (tidx < 16) {
        sharr[tidx] += sharr[tidx + 8];
    }
    __syncwarp();
    if (tidx < 16) {
        sharr[tidx] += sharr[tidx + 4];
    }
    __syncwarp();
    if (tidx < 16) {
        sharr[tidx] += sharr[tidx + 2];
    }
    __syncwarp();
    if (tidx < 16) {
        sharr[tidx] += sharr[tidx + 1];
    }
    __syncwarp();
}

template <typename scalar_t>
__device__ inline void gather_warp_execution_results(scalar_t* sharr, const uint16_t tidx) {
    sharr[tidx] = tidx * CUDA_WARP_SIZE < CUDA_MAX_NUM_THREADS_PER_BLOCK ? sharr[tidx * CUDA_WARP_SIZE] : static_cast<scalar_t>(0.0);
}


// Reduces the contents of a shared memory array of CUDA_MAX_NUM_THREADS_PER_BLOCK using
// warp-powered reduction. The final sum will be stored in the 0-th element of the shared memory array.
template <typename scalar_t>
__device__ void reduce_in_block_using_warp_sums(scalar_t* __restrict__ sh_mem,
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


template <typename scalar_t>
__device__ void reduce_with_shared_memory(
        scalar_t* __restrict__ sh_arr,
        scalar_t current_thread_sum,
        const uint16_t tidx,
        const uint32_t bidx,
        scalar_t* __restrict__ dev_tmp,
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
        sh_arr[tidx] = tidx < total_number_of_blocks ? dev_tmp[tidx] : static_cast<scalar_t>(0.0);
        reduce_in_block_using_warp_sums(sh_arr, tidx);

        if (tidx == 0) {
            grad[0] = sh_arr[0];
        }
    }
}



// Remove this and other FP16 template specializations once arithmetic operators are implemented in c10
// for volatile c10::Half

__device__ void reduce_in_block_using_warp_sums_with_explicit_sync(c10::Half* __restrict__ sh_mem,
        uint16_t tidx) {
    __syncthreads();
    sum_warp_with_explicit_sync(sh_mem + (tidx & ~(CUDA_WARP_SIZE - 1)));

    __syncthreads();
    if (tidx < CUDA_MAX_WARPS_PER_BLOCK) {
        gather_warp_execution_results(sh_mem, tidx);
        sum_warp_with_explicit_sync(sh_mem);
    }

}

template <>
__device__ void reduce_with_shared_memory<c10::Half>(
        c10::Half* __restrict__ sh_arr,
        c10::Half sum,
        const uint16_t tidx,
        const uint32_t bidx,
        c10::Half* __restrict__ dev_tmp,
        int32_t* __restrict__ dev_last_block_counter,
        c10::Half* __restrict__ grad,
        uint32_t total_number_of_blocks) {
    sh_arr[tidx] = sum;

    reduce_in_block_using_warp_sums_with_explicit_sync(sh_arr, tidx);

    if (tidx == 0) {
        dev_tmp[bidx] = sh_arr[0];
    }

    if (last_block(dev_last_block_counter, total_number_of_blocks)) {
        sh_arr[tidx] = tidx < gridDim.x ? dev_tmp[tidx] : static_cast<c10::Half>(0.0);

        reduce_in_block_using_warp_sums_with_explicit_sync(sh_arr, tidx);

        if (tidx == 0) {
            grad[0] = sh_arr[0];
        }
    }
}


#endif // _COMMON_CUDA_FUNCS_CUH_
