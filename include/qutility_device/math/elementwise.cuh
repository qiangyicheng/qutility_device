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
            template <typename ValT = double, typename IntT = int>
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
            /// @param dst dst is interpretted as complex number, so that two times of storage is required
            template <typename ValT = double, typename IntT = int>
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
            template <typename ValT = double, typename IntT = int>
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

            /// @brief Copy one array to another. Only 1D grid with 1D block is allowed
            /// @tparam ValT the type of the data
            /// @tparam IntT the type of the counter. Note that usually int is sufficient and deliver slightly higher performance than std::size_t
            template <typename ValT = double, typename IntT = int>
            __global__ void array_copy(IntT single_size, ValT *dst, const ValT *src)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    dst[itr] = src[itr];
                }
            }

            /// @brief Multiple one array to self. Only 1D grid with 1D block is allowed
            /// @tparam ValT the type of the data
            /// @tparam IntT the type of the counter. Note that usually int is sufficient and deliver slightly higher performance than std::size_t
            template <typename ValT = double, typename IntT = int>
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
            template <typename ValT = double, typename IntT = int>
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
            template <typename ValT = double, typename IntT = int>
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
            template <typename ValT = double, typename IntT = int>
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

            /// @brief Divide array by normalizer on device. Only 1D grid with 1D block is allowed
            /// @tparam ValT the type of the data
            /// @tparam IntT the type of the counter. Note that usually int is sufficient and deliver slightly higher performance than std::size_t
            template <typename ValT = double, typename IntT = int>
            __global__ void array_normalize_device(IntT single_size, ValT *dst, const ValT *normalizer, ValT factor)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                ValT combined_factor = factor / normalizer[0];
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    dst[itr] *= combined_factor;
                }
            }

            /// @brief accumate arrays
            template <typename ValT = double, typename IntT = int>
            __global__ void array_weighted_sum(IntT single_size, IntT n_sum, ValT *dst, const ValT *src, const ValT *coef)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    ValT val = 0.;
                    for (IntT itr_sum = 0; itr_sum < n_sum; ++itr_sum)
                    {
                        val += src[itr_sum * single_size + itr] * coef[itr_sum];
                    }
                    dst[itr] = val;
                }
            }

            /// @brief mix three arrays
            template <typename ValT = double, typename IntT = int>
            __global__ void array_mix_3(IntT single_size, IntT n_mix, ValT *dst, const ValT *src1, const ValT *src2, const ValT *src3, const ValT *coef, ValT factor1, ValT factor2, ValT factor3)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    auto val1 = src1[itr] * factor1;
                    auto val2 = src2[itr] * factor2;
                    auto val3 = src3[itr] * factor3;
                    for (IntT itr_mix = 0; itr_mix < n_mix; ++itr_mix)
                    {
                        dst[itr_mix * single_size + itr] = val1 * coef[itr_mix * 3 + 0] + val2 * coef[itr_mix * 3 + 1] + val3 * coef[itr_mix * 3 + 2];
                    }
                }
            }

            /// @brief mix two arrays
            template <typename ValT = double, typename IntT = int>
            __global__ void array_mix_2(IntT single_size, IntT n_mix, ValT *dst, const ValT *src1, const ValT *src2, const ValT *coef, ValT factor1, ValT factor2)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    auto val1 = src1[itr] * factor1;
                    auto val2 = src2[itr] * factor2;
                    for (IntT itr_mix = 0; itr_mix < n_mix; ++itr_mix)
                    {
                        dst[itr_mix * single_size + itr] = val1 * coef[itr_mix * 2 + 0] + val2 * coef[itr_mix * 2 + 1];
                    }
                }
            }

            /// @brief mix one array(s)
            template <typename ValT = double, typename IntT = int>
            __global__ void array_mix_1(IntT single_size, IntT n_mix, ValT *dst, const ValT *src1, const ValT *coef, ValT factor1)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    auto val1 = src1[itr] * factor1;
                    for (IntT itr_mix = 0; itr_mix < n_mix; ++itr_mix)
                    {
                        dst[itr_mix * single_size + itr] = val1 * coef[itr_mix * 2 + 0];
                    }
                }
            }

            template <typename ValT = double, typename IntT = int>
            __global__ void scft_calc_new_field_2(IntT single_size, ValT *wA, ValT *wB, const ValT *phiA, const ValT *phiB, ValT xN, ValT acceptance)
            {
                static_assert(std::is_integral<IntT>::value, "Only integer allowed here");
                IntT thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                IntT grid_size = gridDim.x * blockDim.x;
                for (IntT itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    auto ksi = 0.5 * (wA[itr] + wB[itr] - xN);
                    wA[itr] += acceptance * (xN * phiB[itr] + ksi - wA[itr]);
                    wB[itr] += acceptance * (xN * phiA[itr] + ksi - wB[itr]);
                }
            }

        }
    }
}