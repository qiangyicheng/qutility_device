#pragma once

#include <type_traits>
#include <cstddef>

#include "device_api/device_api_cuda_runtime.h"
#include "device_api/device_api_cuda_device.h"

#include "qutility_device/sync_grid.cuh"

#include "common.cuh"

namespace qutility
{
    namespace device
    {
        namespace math
        {

            /// @brief Only 1D grid with 1D block is allowed. Number of blocks mush be larger than dimensionality
            ///        The resulting data makes use of the Hermitian symmetry at the last dimension
            template <bool IfLastDimModified = false, typename ValT = double, typename IntT = int>
            __global__ __forceinline__ void laplacian_eigenvalue_1D(ValT *k, ValT factor_x, IntT Nx)
            {
                QUTILITY_DEVICE_SYNC_GRID_PREPARE;

                const IntT threads_per_block = blockDim.x;
                const IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                const IntT grid_size = gridDim.x * blockDim.x;

                const IntT Nx_hermit = Nx / 2 + 1;

                for (IntT itr_x = thread_rank; itr_x < Nx_hermit; itr_x += grid_size)
                {
                    auto val = itr_x * itr_x * factor_x;
                    if constexpr (IfLastDimModified)
                        val *= (2. - ((itr_x % ((Nx + 1) / 2)) == 0));

                    k[itr_x] = val;
                }
            }

            /// @brief Only 1D grid with 1D block is allowed. Number of blocks mush be larger than dimensionality
            ///        The resulting data makes use of the Hermitian symmetry at the last dimension
            template <bool IfLastDimModified = false, typename ValT = double, typename IntT = int>
            __global__ __forceinline__ void laplacian_eigenvalue_2D(ValT *k, ValT *working, ValT factor_x, ValT factor_y, IntT Nx, IntT Ny)
            {
                QUTILITY_DEVICE_SYNC_GRID_PREPARE;

                const IntT threads_per_block = blockDim.x;
                const IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                const IntT block_per_dim = gridDim.x / 2;
                const IntT Ny_hermit = Ny / 2 + 1;

                // blocks for eigenvalues on x direction
                const IntT required_block_for_x = (Nx + threads_per_block - 1) / threads_per_block;
                const IntT block_for_x = required_block_for_x > block_per_dim ? block_per_dim : required_block_for_x;
                const IntT thread_for_x = block_for_x * threads_per_block;
                const IntT shift_for_x = 0;
                auto k_x = working + shift_for_x;

                // blocks for eigenvalues on y direction
                const IntT required_block_for_y = (Ny_hermit + threads_per_block - 1) / threads_per_block;
                const IntT block_for_y = required_block_for_y > block_per_dim ? block_per_dim : required_block_for_y;
                const IntT thread_for_y = block_for_y * threads_per_block;
                const IntT shift_for_y = utility::next_pow_2(Nx);
                auto k_y = working + shift_for_y;

                if (blockIdx.x < block_for_x)
                {
                    for (IntT itr_x = thread_rank; itr_x < Nx; itr_x += thread_for_x)
                    {
                        k_x[itr_x] = (ValT)itr_x - (int)(itr_x > ((Nx + 1) / 2)) * (ValT)Nx;
                        k_x[itr_x] *= k_x[itr_x];
                        k_x[itr_x] *= factor_x;
                    }
                }
                else if (blockIdx.x < block_for_x + block_for_y)
                {
                    for (IntT itr_y = thread_rank - thread_for_x; itr_y < Ny_hermit; itr_y += thread_for_y)
                    {
                        k_y[itr_y] = itr_y * itr_y * factor_y;
                    }
                }

                QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));

                for (IntT itr_x = blockIdx.x; itr_x < Nx; itr_x += gridDim.x)
                {
                    for (IntT itr_y = threadIdx.x; itr_y < Ny_hermit; itr_y += blockDim.x)
                    {
                        auto val = k_x[itr_x] + k_y[itr_y];
                        if constexpr (IfLastDimModified)
                            val *= (2. - ((itr_y % ((Ny + 1) / 2)) == 0));

                        k[itr_x * Ny_hermit + itr_y] = val;
                    }
                }
            }

            /// @brief Only 1D grid with 1D block is allowed. Number of blocks mush be larger than dimensionality
            ///        The resulting data makes use of the Hermitian symmetry at the last dimension
            template <bool IfLastDimModified = false, typename ValT = double, typename IntT = int>
            __global__ __forceinline__ void laplacian_eigenvalue_3D(ValT *k, ValT *working, ValT factor_x, ValT factor_y, ValT factor_z, IntT Nx, IntT Ny, IntT Nz)
            {
                QUTILITY_DEVICE_SYNC_GRID_PREPARE;

                const IntT threads_per_block = blockDim.x;
                const IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                const IntT block_per_dim = gridDim.x / 2;
                const IntT Nz_hermit = Nz / 2 + 1;

                // blocks for eigenvalues on x direction
                const IntT required_block_for_x = (Nx + threads_per_block - 1) / threads_per_block;
                const IntT block_for_x = required_block_for_x > block_per_dim ? block_per_dim : required_block_for_x;
                const IntT thread_for_x = block_for_x * threads_per_block;
                const IntT shift_for_x = 0;
                auto k_x = working + shift_for_x;

                // blocks for eigenvalues on y direction
                const IntT required_block_for_y = (Ny + threads_per_block - 1) / threads_per_block;
                const IntT block_for_y = required_block_for_y > block_per_dim ? block_per_dim : required_block_for_y;
                const IntT thread_for_y = block_for_y * threads_per_block;
                const IntT shift_for_y = utility::next_pow_2(Nx);
                auto k_y = working + shift_for_y;

                // blocks for eigenvalues on z direction
                const IntT required_block_for_z = (Nz_hermit + threads_per_block - 1) / threads_per_block;
                const IntT block_for_z = required_block_for_z > block_per_dim ? block_per_dim : required_block_for_z;
                const IntT thread_for_z = block_for_z * threads_per_block;
                const IntT shift_for_z = shift_for_y + utility::next_pow_2(Ny);
                auto k_z = working + shift_for_z;

                if (blockIdx.x < block_for_x)
                {
                    for (IntT itr_x = thread_rank; itr_x < Nx; itr_x += thread_for_x)
                    {
                        k_x[itr_x] = (ValT)itr_x - (int)(itr_x > ((Nx + 1) / 2)) * (ValT)Nx;
                        k_x[itr_x] *= k_x[itr_x];
                        k_x[itr_x] *= factor_x;
                    }
                }
                else if (blockIdx.x < block_for_x + block_for_y)
                {
                    for (IntT itr_y = thread_rank - thread_for_x; itr_y < Ny; itr_y += thread_for_y)
                    {
                        k_y[itr_y] = (ValT)itr_y - (int)(itr_y > ((Ny + 1) / 2)) * (ValT)Ny;
                        k_y[itr_y] *= k_y[itr_y];
                        k_y[itr_y] *= factor_y;
                    }
                }
                else if (blockIdx.x < block_for_x + block_for_y + block_for_z)
                {
                    for (IntT itr_z = thread_rank - thread_for_x - thread_for_y; itr_z < Nz_hermit; itr_z += thread_for_z)
                    {
                        k_z[itr_z] = itr_z * itr_z * factor_z;
                    }
                }

                QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));

                for (IntT itr_block = blockIdx.x; itr_block < Nx * Ny; itr_block += gridDim.x)
                {
                    __shared__ ValT val_xy;
                    if (threadIdx.x == 0)
                        val_xy = k_x[itr_block / Ny] + k_y[itr_block % Ny];
                    dapi___syncthreads();
                    for (IntT itr_z = threadIdx.x; itr_z < Nz_hermit; itr_z += blockDim.x)
                    {
                        auto val = val_xy + k_z[itr_z];
                        if constexpr (IfLastDimModified)
                            val *= (2. - ((itr_z % ((Nz + 1) / 2)) == 0));

                        k[itr_block * Nz_hermit + itr_z] = val;
                    }
                }
            }

        }
    }
}