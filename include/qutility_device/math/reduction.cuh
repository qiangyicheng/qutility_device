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

            template <std::size_t ThreadsPerBlock = 256, typename ValT = double, typename IntT = int>
            __global__ void scft_calc_new_field_and_energy_2(
                IntT single_size,
                ValT *FEValues, ValT *wA, ValT *wB,
                const ValT *Q, const ValT *phiA, const ValT *phiB,
                ValT *working,
                ValT xN, ValT acceptance)
            {
                QUTILITY_DEVICE_SYNC_GRID_PREPARE;

                const IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                const IntT grid_size = gridDim.x * blockDim.x;

                using BlockReduceT = dapi_cub::BlockReduce<ValT, ThreadsPerBlock, dapi_cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>;
                __shared__ typename BlockReduceT::TempStorage temp_storage;

                ValT phiphi = 0.0;
                ValT phiw = 0.0;
                ValT ww = 0.0;
                ValT wdiffwdiff = 0.0;
                ValT maxincomp = 0.0;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    auto ksi = 0.5 * (wA[itr] + wB[itr] - xN);
                    auto wAdiff = xN * phiB[itr] + ksi - wA[itr];
                    auto wBdiff = xN * phiA[itr] + ksi - wB[itr];
                    ww += wA[itr] * wA[itr] + wB[itr] * wB[itr];
                    wdiffwdiff += wAdiff * wAdiff + wBdiff * wBdiff;
                    phiphi += phiA[itr] * phiB[itr];
                    auto incomp = 1. - phiA[itr] - phiB[itr];
                    phiw += phiA[itr] * wA[itr] + phiB[itr] * wB[itr] + ksi * incomp;
                    wA[itr] += acceptance * wAdiff;
                    wB[itr] += acceptance * wBdiff;
                    incomp = fabs(incomp);
                    maxincomp = maxincomp > incomp ? maxincomp : incomp;
                }
                dapi___syncthreads();
                {
                    ValT phiphians = BlockReduceT(temp_storage).Sum(phiphi);
                    if (threadIdx.x == 0)
                        working[blockIdx.x] = phiphians;
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                    if (thread_rank == 0)
                    {
                        phiphians = 0;
                        for (IntT block = 0; block < gridDim.x; block++)
                            phiphians += working[block];
                        FEValues[1] = xN * phiphians / single_size;
                    }
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                }
                {
                    ValT phiwans = BlockReduceT(temp_storage).Sum(phiw);
                    if (threadIdx.x == 0)
                        working[blockIdx.x] = phiwans;
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                    if (thread_rank == 0)
                    {
                        phiwans = 0;
                        for (IntT block = 0; block < gridDim.x; block++)
                            phiwans += working[block];
                        FEValues[3] = -phiwans / single_size;
                    }
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                }
                {
                    ValT wwans = BlockReduceT(temp_storage).Sum(ww);
                    ValT wdiffwdiffans = BlockReduceT(temp_storage).Sum(wdiffwdiff);
                    if (threadIdx.x == 0)
                    {
                        working[blockIdx.x] = wwans;
                    }
                    if (threadIdx.x == 1)
                    {
                        working[gridDim.x + blockIdx.x] = wdiffwdiffans;
                    }
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                    if (thread_rank == 0)
                    {
                        wwans = 0;
                        wdiffwdiffans = 0;
                        for (IntT block = 0; block < gridDim.x; block++)
                        {
                            wwans += working[block];
                            wdiffwdiffans += working[gridDim.x + block];
                        }
                        FEValues[6] = wdiffwdiffans / wwans;
                    }
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                }
                {
                    ValT maxincompans = BlockReduceT(temp_storage).Reduce(maxincomp, dapi_cub::Max());
                    if (threadIdx.x == 0)
                        working[blockIdx.x] = maxincompans;
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                    if (thread_rank == 0)
                    {
                        maxincompans = 0;
                        for (IntT block = 0; block < gridDim.x; block++)
                            maxincompans = maxincompans > working[block] ? maxincompans : working[block];
                        FEValues[5] = maxincompans;
                        FEValues[2] = -log(Q[0]);
                        FEValues[4] = 0.;
                    }
                }
            }

            template <std::size_t ThreadsPerBlock = 256, typename ValT = double, typename IntT = int>
            __global__ void scft_calc_field_error_and_energy_2(
                IntT single_size,
                ValT *FEValues, ValT *wA_diff, ValT *wB_diff,
                const ValT *Q, const ValT *wA, const ValT *wB, const ValT *phiA, const ValT *phiB,
                ValT *working,
                ValT xN, ValT acceptance)
            {
                QUTILITY_DEVICE_SYNC_GRID_PREPARE;

                const IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                const IntT grid_size = gridDim.x * blockDim.x;

                using BlockReduceT = dapi_cub::BlockReduce<ValT, ThreadsPerBlock, dapi_cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>;
                __shared__ typename BlockReduceT::TempStorage temp_storage;

                ValT phiphi = 0.0;
                ValT phiw = 0.0;
                ValT ww = 0.0;
                ValT wdiffwdiff = 0.0;
                ValT maxincomp = 0.0;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    auto ksi = 0.5 * (wA[itr] + wB[itr] - xN);
                    auto wAdiff = xN * phiB[itr] + ksi - wA[itr];
                    auto wBdiff = xN * phiA[itr] + ksi - wB[itr];
                    ww += wA[itr] * wA[itr] + wB[itr] * wB[itr];
                    wdiffwdiff += wAdiff * wAdiff + wBdiff * wBdiff;
                    phiphi += phiA[itr] * phiB[itr];
                    auto incomp = 1. - phiA[itr] - phiB[itr];
                    phiw += phiA[itr] * wA[itr] + phiB[itr] * wB[itr] + ksi * incomp;
                    wA_diff[itr] = acceptance * wAdiff;
                    wB_diff[itr] = acceptance * wBdiff;
                    incomp = fabs(incomp);
                    maxincomp = maxincomp > incomp ? maxincomp : incomp;
                }
                dapi___syncthreads();
                {
                    ValT phiphians = BlockReduceT(temp_storage).Sum(phiphi);
                    if (threadIdx.x == 0)
                        working[blockIdx.x] = phiphians;
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                    if (thread_rank == 0)
                    {
                        phiphians = 0;
                        for (IntT block = 0; block < gridDim.x; block++)
                            phiphians += working[block];
                        FEValues[1] = xN * phiphians / single_size;
                    }
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                }
                {
                    ValT phiwans = BlockReduceT(temp_storage).Sum(phiw);
                    if (threadIdx.x == 0)
                        working[blockIdx.x] = phiwans;
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                    if (thread_rank == 0)
                    {
                        phiwans = 0;
                        for (IntT block = 0; block < gridDim.x; block++)
                            phiwans += working[block];
                        FEValues[3] = -phiwans / single_size;
                    }
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                }
                {
                    ValT wwans = BlockReduceT(temp_storage).Sum(ww);
                    ValT wdiffwdiffans = BlockReduceT(temp_storage).Sum(wdiffwdiff);
                    if (threadIdx.x == 0)
                    {
                        working[blockIdx.x] = wwans;
                    }
                    if (threadIdx.x == 1)
                    {
                        working[gridDim.x + blockIdx.x] = wdiffwdiffans;
                    }
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                    if (thread_rank == 0)
                    {
                        wwans = 0;
                        wdiffwdiffans = 0;
                        for (IntT block = 0; block < gridDim.x; block++)
                        {
                            wwans += working[block];
                            wdiffwdiffans += working[gridDim.x + block];
                        }
                        FEValues[6] = wdiffwdiffans / wwans;
                    }
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                }
                {
                    ValT maxincompans = BlockReduceT(temp_storage).Reduce(maxincomp, dapi_cub::Max());
                    if (threadIdx.x == 0)
                        working[blockIdx.x] = maxincompans;
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                    if (thread_rank == 0)
                    {
                        maxincompans = 0;
                        for (IntT block = 0; block < gridDim.x; block++)
                            maxincompans = maxincompans > working[block] ? maxincompans : working[block];
                        FEValues[5] = maxincompans;
                        FEValues[2] = -log(Q[0]);
                        FEValues[4] = 0.;
                    }
                }
            }

        }
    }
}
