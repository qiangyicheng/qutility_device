#pragma once

#include <type_traits>
#include <cstddef>

#include "device_api/device_api_cuda_runtime.h"
#include "device_api/device_api_cuda_device.h"
#include "device_api/device_api_cub.h"

#include "qutility_device/sync_grid.cuh"

namespace qutility
{
    namespace device
    {
        namespace math
        {
            template <std::size_t ThreadsPerBlock = 256, typename ValT = double, typename IntT = int>
            __global__ void array_mean_and_normalize(IntT single_size, ValT *srcdst, ValT *mean, ValT *working, ValT factor_mean, ValT factor_normalize)
            {
                QUTILITY_DEVICE_SYNC_GRID_PREPARE;

                const IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                const IntT grid_size = gridDim.x * blockDim.x;

                using BlockReduceT = dapi_cub::BlockReduce<ValT, ThreadsPerBlock, dapi_cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>;
                __shared__ typename BlockReduceT::TempStorage temp_storage;
                {
                    ValT data = 0.0;
                    for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                    {
                        data += srcdst[itr];
                    }
                    dapi___syncthreads();
                    ValT agg = BlockReduceT(temp_storage).Sum(data);
                    if (threadIdx.x == 0)
                    {
                        working[blockIdx.x] = agg;
                    }
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                    if (thread_rank == 0)
                    {
                        for (IntT block = 1; block < gridDim.x; block++)
                        {
                            working[0] += working[block];
                        }
                        mean[0] = working[0] * factor_mean / single_size;
                    }
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                }
                ValT factor = (single_size * factor_normalize) / working[0];
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    srcdst[itr] *= factor;
                }
            }

            template <std::size_t ThreadsPerBlock = 256, typename ValT = double, typename IntT = int>
            __global__ void array_remove_mean(IntT single_size, ValT *srcdst, ValT *working)
            {
                QUTILITY_DEVICE_SYNC_GRID_PREPARE;

                const IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                const IntT grid_size = gridDim.x * blockDim.x;

                using BlockReduceT = dapi_cub::BlockReduce<ValT, ThreadsPerBlock, dapi_cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>;
                __shared__ typename BlockReduceT::TempStorage temp_storage;
                {
                    ValT data = 0.0;
                    for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                    {
                        data += srcdst[itr];
                    }
                    dapi___syncthreads();
                    ValT agg = BlockReduceT(temp_storage).Sum(data);
                    if (threadIdx.x == 0)
                    {
                        working[blockIdx.x] = agg;
                    }
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                    if (thread_rank == 0)
                    {
                        for (IntT block = 1; block < gridDim.x; block++)
                        {
                            working[0] += working[block];
                        }
                        working[0] /= single_size;
                    }
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                }
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    srcdst[itr] -= working[0];
                }
            }
        }
    }
}
