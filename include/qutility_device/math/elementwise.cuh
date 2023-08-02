#pragma once

#include <type_traits>
#include <cstddef>

#include "device_api/device_api_cuda_runtime.h"
#include "device_api/device_api_cuda_device.h"

namespace qutility
{
    namespace device
    {
        namespace math
        {

            /// @brief Calculate the exp of a single field stored in src, and copy the result to dst for dup times. Only 1D grid with 1D block is allowed
            /// @tparam ValT the type of the data
            /// @tparam IntT the type of the counter. Note that usually int is sufficient and deliver slightly higher performance than std::size_t
            template <typename ValT, typename IntT = int>
            __global__ void array_exp_dup(IntT single_size, IntT dup, ValT *dst, const ValT *src, ValT coef, ValT index)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    ValT val;
                    if constexpr (std::is_same<ValT, double>::value)
                    {
                        val = coef * exp(index * src[itr]);
                    }
                    else if constexpr (std::is_same<ValT, float>::value)
                    {
                        val = coef * expf(index * src[itr]);
                    }
                    else
                    {
                        static_assert(std::is_floating_point<ValT>::value, "Only float point allowed here");
                    }
                    for (IntT itr_dup = 0; itr_dup < dup; ++itr_dup)
                    {
                        dst[itr_dup * single_size + itr] = val;
                    }
                }
            }

            /// @brief Calculate the exp of an imaginary single field stored in src, and copy the result to dst for dup times. Only 1D grid with 1D block is allowed
            /// @tparam ValT the type of the data
            /// @tparam IntT the type of the counter. Note that usually int is sufficient and deliver slightly higher performance than std::size_t
            template <typename ValT, typename IntT = int>
            __global__ void array_imaginary_exp_dup(IntT single_size, IntT dup, ValT *dst, const ValT *src, ValT coef, ValT index)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    ValT sine;
                    ValT cosine;
                    if constexpr (std::is_same<ValT, double>::value)
                    {
                        sincos(index * src[itr], &sine, &cosine);
                    }
                    else if constexpr (std::is_same<ValT, float>::value)
                    {
                        sincosf(index * src[itr], &sine, &cosine);
                    }
                    else
                    {
                        static_assert(std::is_floating_point<ValT>::value, "Only float point allowed here");
                    }
                    sine *= coef;
                    cosine *= coef;
                    for (IntT itr_dup = 0; itr_dup < dup; ++itr_dup)
                    {
                        dst[2 * (itr_dup * single_size + itr) + 0] = cosine;
                        dst[2 * (itr_dup * single_size + itr) + 1] = sine;
                    }
                }
            }

            /// @brief Add one array to self. Only 1D grid with 1D block is allowed
            /// @tparam ValT the type of the data
            /// @tparam IntT the type of the counter. Note that usually int is sufficient and deliver slightly higher performance than std::size_t
            template <typename ValT, typename IntT = int>
            __global__ void array_selfadd(IntT single_size, ValT *dst, const ValT *src)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    dst[itr] += src[itr];
                }
            }

            /// @brief Multiple one array to self. Only 1D grid with 1D block is allowed
            /// @tparam ValT the type of the data
            /// @tparam IntT the type of the counter. Note that usually int is sufficient and deliver slightly higher performance than std::size_t
            template <typename ValT, typename IntT = int>
            __global__ void array_selfmul(IntT single_size, ValT *dst, const ValT *src)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    dst[itr] *= src[itr];
                }
            }

            /// @brief Multiple two arrays. Only 1D grid with 1D block is allowed
            /// @tparam ValT the type of the data
            /// @tparam IntT the type of the counter. Note that usually int is sufficient and deliver slightly higher performance than std::size_t
            template <typename ValT, typename IntT = int>
            __global__ void array_mul(IntT single_size, ValT *dst, const ValT *src1, const ValT *src2)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    dst[itr] = src1[itr] * src2[itr];
                }
            }

            /// @brief Set array by val. Only 1D grid with 1D block is allowed
            /// @tparam ValT the type of the data
            /// @tparam IntT the type of the counter. Note that usually int is sufficient and deliver slightly higher performance than std::size_t
            template <typename ValT, typename IntT = int>
            __global__ void array_set(IntT single_size, ValT *dst, ValT val)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    dst[itr] = val;
                }
            }

            /// @brief Scale array by val. Only 1D grid with 1D block is allowed
            /// @tparam ValT the type of the data
            /// @tparam IntT the type of the counter. Note that usually int is sufficient and deliver slightly higher performance than std::size_t
            template <typename ValT, typename IntT = int>
            __global__ void array_scale(IntT single_size, ValT *dst, ValT val)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    dst[itr] *= val;
                }
            }

        }
    }
}