#pragma once

#include "qutility/message.h"

#include "thrust/device_vector.h"

#include <tuple>
#include <map>
#include <vector>
#include <boost/unordered/unordered_map.hpp>

#include "device_api/device_api_cufft.h"

namespace qutility
{
    namespace device
    {
        namespace fft_cb
        {
            namespace kernel
            {
                ////////////////////////////////////////////////////////////////////////////////
                // CallBack Function
                ////////////////////////////////////////////////////////////////////////////////
                struct D2ZLoadSideLoadInfo_st
                {
                    const dapi_cufftDoubleReal *m;
                };
                using D2ZLoadSideLoadInfo_t = D2ZLoadSideLoadInfo_st *;
                __device__ __forceinline__ dapi_cufftDoubleReal D2ZLoadSideLoadImpl(const dapi_cufftDoubleReal *dataIn, size_t offset, D2ZLoadSideLoadInfo_t data, void *sharedPointer = nullptr)
                {
                    return data->m[offset];
                }
                __device__ __forceinline__ dapi_cufftDoubleReal D2ZLoadSideLoad(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer)
                {
                    return D2ZLoadSideLoadImpl((dapi_cufftDoubleReal *)dataIn, offset, (D2ZLoadSideLoadInfo_t)callerInfo);
                }

                extern __device__ dapi_cufftCallbackLoadD D2ZLoadSideLoadDevicePtr;

                ////////////////////////////////////////////////////////////////////////////////
                // CallBack Function
                ////////////////////////////////////////////////////////////////////////////////
                struct D2ZLoadMulInfo_st
                {
                    const dapi_cufftDoubleReal *m;
                };
                using D2ZLoadMulInfo_t = D2ZLoadMulInfo_st *;
                __device__ __forceinline__ dapi_cufftDoubleReal D2ZLoadMulImpl(const dapi_cufftDoubleReal *dataIn, size_t offset, D2ZLoadMulInfo_t data, void *sharedPointer = nullptr)
                {
                    return dataIn[offset] * data->m[offset];
                }
                __device__ __forceinline__ dapi_cufftDoubleReal D2ZLoadMul(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer)
                {
                    return D2ZLoadMulImpl((dapi_cufftDoubleReal *)dataIn, offset, (D2ZLoadMulInfo_t)callerInfo);
                }

                extern __device__ dapi_cufftCallbackLoadD D2ZLoadMulDevicePtr;

                ////////////////////////////////////////////////////////////////////////////////
                // CallBack Function
                ////////////////////////////////////////////////////////////////////////////////
                struct D2ZLoadShiftLoadMulInfo_st
                {
                    const dapi_cufftDoubleReal *m;
                    const std::ptrdiff_t shift;
                };
                using D2ZLoadShiftLoadMulInfo_t = D2ZLoadShiftLoadMulInfo_st *;
                __device__ __forceinline__ dapi_cufftDoubleReal D2ZLoadShiftLoadMulImpl(const dapi_cufftDoubleReal *dataIn, size_t offset, D2ZLoadShiftLoadMulInfo_t data, void *sharedPointer = nullptr)
                {
                    return (dataIn + data->shift)[offset] * data->m[offset];
                }
                __device__ __forceinline__ dapi_cufftDoubleReal D2ZLoadShiftLoadMul(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer)
                {
                    return D2ZLoadShiftLoadMulImpl((dapi_cufftDoubleReal *)dataIn, offset, (D2ZLoadShiftLoadMulInfo_t)callerInfo);
                }

                extern __device__ dapi_cufftCallbackLoadD D2ZLoadShiftLoadMulDevicePtr;

                ////////////////////////////////////////////////////////////////////////////////
                // CallBack Function
                ////////////////////////////////////////////////////////////////////////////////
                struct D2ZLoadHalfMulIncreMulInfo_st
                {
                    const dapi_cufftDoubleReal *m;
                    const dapi_cufftDoubleReal *incre_mul_src;
                    dapi_cufftDoubleReal *incre_dst;
                };
                using D2ZLoadHalfMulIncreMulInfo_t = D2ZLoadHalfMulIncreMulInfo_st *;
                __device__ __forceinline__ dapi_cufftDoubleReal D2ZLoadHalfMulIncreMulImpl(const dapi_cufftDoubleReal *dataIn, size_t offset, D2ZLoadHalfMulIncreMulInfo_t data, void *sharedPointer = nullptr)
                {
                    auto val = dataIn[offset];
                    data->incre_dst[offset] += 0.5 * data->incre_mul_src[offset] * val;
                    return val * data->m[offset];
                }
                __device__ __forceinline__ dapi_cufftDoubleReal D2ZLoadHalfMulIncreMul(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer)
                {
                    return D2ZLoadHalfMulIncreMulImpl((dapi_cufftDoubleReal *)dataIn, offset, (D2ZLoadHalfMulIncreMulInfo_t)callerInfo);
                }

                extern __device__ dapi_cufftCallbackLoadD D2ZLoadHalfMulIncreMulDevicePtr;

                ////////////////////////////////////////////////////////////////////////////////
                // CallBack Function
                ////////////////////////////////////////////////////////////////////////////////
                struct D2ZStoreMulRealInfo_st
                {
                    const dapi_cufftDoubleReal *m;
                };
                using D2ZStoreMulRealInfo_t = D2ZStoreMulRealInfo_st *;
                __device__ __forceinline__ void D2ZStoreMulRealImpl(dapi_cufftDoubleComplex *dataOut, size_t offset, dapi_cufftDoubleComplex element, D2ZStoreMulRealInfo_t data, void *sharedPointer = nullptr)
                {
                    dataOut[offset].x = element.x * data->m[offset];
                    dataOut[offset].y = element.y * data->m[offset];
                }
                __device__ __forceinline__ void D2ZStoreMulReal(void *dataOut, size_t offset, dapi_cufftDoubleComplex element, void *callerInfo, void *sharedPointer)
                {
                    return D2ZStoreMulRealImpl((dapi_cufftDoubleComplex *)dataOut, offset, element, (D2ZStoreMulRealInfo_t)callerInfo);
                }
                extern __device__ dapi_cufftCallbackStoreZ D2ZStoreMulRealDevicePtr;

                ////////////////////////////////////////////////////////////////////////////////
                // CallBack Function
                ////////////////////////////////////////////////////////////////////////////////
                struct D2ZStoreMulComplexDropRealIncreInfo_st
                {
                    const dapi_cufftDoubleComplex *incre_mul_src;
                    dapi_cufftDoubleReal *incre_dst;
                };
                using D2ZStoreMulComplexDropRealIncreInfo_t = D2ZStoreMulComplexDropRealIncreInfo_st *;
                __device__ __forceinline__ void D2ZStoreMulComplexDropRealIncreImpl(dapi_cufftDoubleComplex *dataOut, size_t offset, dapi_cufftDoubleComplex element, D2ZStoreMulComplexDropRealIncreInfo_t data, void *sharedPointer = nullptr)
                {
                    data->incre_dst[offset] += element.x * data->incre_mul_src[offset].y - element.y * data->incre_mul_src[offset].x;
                }
                __device__ __forceinline__ void D2ZStoreMulComplexDropRealIncre(void *dataOut, size_t offset, dapi_cufftDoubleComplex element, void *callerInfo, void *sharedPointer)
                {
                    return D2ZStoreMulComplexDropRealIncreImpl((dapi_cufftDoubleComplex *)dataOut, offset, element, (D2ZStoreMulComplexDropRealIncreInfo_t)callerInfo);
                }
                extern __device__ dapi_cufftCallbackStoreZ D2ZStoreMulComplexDropRealIncreDevicePtr;

                ////////////////////////////////////////////////////////////////////////////////
                // CallBack Function
                ////////////////////////////////////////////////////////////////////////////////
                struct D2ZStoreHalfMulComplexDropRealIncreInfo_st
                {
                    const dapi_cufftDoubleComplex *incre_mul_src;
                    dapi_cufftDoubleReal *incre_dst;
                };
                using D2ZStoreHalfMulComplexDropRealIncreInfo_t = D2ZStoreHalfMulComplexDropRealIncreInfo_st *;
                __device__ __forceinline__ void D2ZStoreHalfMulComplexDropRealIncreImpl(dapi_cufftDoubleComplex *dataOut, size_t offset, dapi_cufftDoubleComplex element, D2ZStoreHalfMulComplexDropRealIncreInfo_t data, void *sharedPointer = nullptr)
                {
                    data->incre_dst[offset] += 0.5 * (element.x * data->incre_mul_src[offset].y - element.y * data->incre_mul_src[offset].x);
                }
                __device__ __forceinline__ void D2ZStoreHalfMulComplexDropRealIncre(void *dataOut, size_t offset, dapi_cufftDoubleComplex element, void *callerInfo, void *sharedPointer)
                {
                    return D2ZStoreHalfMulComplexDropRealIncreImpl((dapi_cufftDoubleComplex *)dataOut, offset, element, (D2ZStoreHalfMulComplexDropRealIncreInfo_t)callerInfo);
                }
                extern __device__ dapi_cufftCallbackStoreZ D2ZStoreHalfMulComplexDropRealIncreDevicePtr;

                ////////////////////////////////////////////////////////////////////////////////
                // CallBack Function
                ////////////////////////////////////////////////////////////////////////////////
                struct Z2DLoadMulImaginaryInfo_st
                {
                    const dapi_cufftDoubleReal *m;
                };
                using Z2DLoadMulImaginaryInfo_t = Z2DLoadMulImaginaryInfo_st *;
                __device__ __forceinline__ dapi_cufftDoubleComplex Z2DLoadMulImaginaryImpl(const dapi_cufftDoubleComplex *dataIn, size_t offset, Z2DLoadMulImaginaryInfo_t data, void *sharedPointer = nullptr)
                {
                    return dapi_cufftDoubleComplex{
                        - dataIn[offset].y * data->m[offset], dataIn[offset].x * data->m[offset]};
                }
                __device__ __forceinline__ dapi_cufftDoubleComplex Z2DLoadMulImaginary(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer)
                {
                    return Z2DLoadMulImaginaryImpl((dapi_cufftDoubleComplex *)dataIn, offset, (Z2DLoadMulImaginaryInfo_t)callerInfo);
                }
                extern __device__ dapi_cufftCallbackLoadZ Z2DLoadMulImaginaryDevicePtr;

                ////////////////////////////////////////////////////////////////////////////////
                // CallBack Function
                ////////////////////////////////////////////////////////////////////////////////
                struct Z2DLoadMulComplexInfo_st
                {
                    const dapi_cufftDoubleComplex *m;
                };
                using Z2DLoadMulComplexInfo_t = Z2DLoadMulComplexInfo_st *;
                __device__ __forceinline__ dapi_cufftDoubleComplex Z2DLoadMulComplexImpl(const dapi_cufftDoubleComplex *dataIn, size_t offset, Z2DLoadMulComplexInfo_t data, void *sharedPointer = nullptr)
                {
                    return dapi_cufftDoubleComplex{
                        dataIn[offset].x * data->m[offset].x - dataIn[offset].y * data->m[offset].y,
                        dataIn[offset].x * data->m[offset].y + dataIn[offset].y * data->m[offset].x};
                }
                __device__ __forceinline__ dapi_cufftDoubleComplex Z2DLoadMulComplex(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer)
                {
                    return Z2DLoadMulComplexImpl((dapi_cufftDoubleComplex *)dataIn, offset, (Z2DLoadMulComplexInfo_t)callerInfo);
                }
                extern __device__ dapi_cufftCallbackLoadZ Z2DLoadMulComplexDevicePtr;

                ////////////////////////////////////////////////////////////////////////////////
                // CallBack Function
                ////////////////////////////////////////////////////////////////////////////////
                struct Z2DStoreMulInfo_st
                {
                    const dapi_cufftDoubleReal *m;
                };
                using Z2DStoreMulInfo_t = Z2DStoreMulInfo_st *;
                __device__ __forceinline__ void Z2DStoreMulImpl(dapi_cufftDoubleReal *dataOut, size_t offset, dapi_cufftDoubleReal element, Z2DStoreMulInfo_t data, void *sharedPointer = nullptr)
                {
                    dataOut[offset] = element * data->m[offset];
                }
                __device__ __forceinline__ void Z2DStoreMul(void *dataOut, size_t offset, dapi_cufftDoubleReal element, void *callerInfo, void *sharedPointer)
                {
                    return Z2DStoreMulImpl((dapi_cufftDoubleReal *)dataOut, offset, element, (Z2DStoreMulInfo_t)callerInfo);
                }
                extern __device__ dapi_cufftCallbackStoreD Z2DStoreMulDevicePtr;
                ////////////////////////////////////////////////////////////////////////////////
                // CallBack Function
                ////////////////////////////////////////////////////////////////////////////////
                struct Z2DStoreMulMulIncreInfo_st
                {
                    const dapi_cufftDoubleReal *m;
                    const dapi_cufftDoubleReal *incre_mul_src;
                    dapi_cufftDoubleReal *incre_dst;
                };
                using Z2DStoreMulMulIncreInfo_t = Z2DStoreMulMulIncreInfo_st *;
                __device__ __forceinline__ void Z2DStoreMulMulIncreImpl(dapi_cufftDoubleReal *dataOut, size_t offset, dapi_cufftDoubleReal element, Z2DStoreMulMulIncreInfo_t data, void *sharedPointer = nullptr)
                {
                    auto val = element * data->m[offset];
                    dataOut[offset] = val;
                    data->incre_dst[offset] += data->incre_mul_src[offset] * val;
                }
                __device__ __forceinline__ void Z2DStoreMulMulIncre(void *dataOut, size_t offset, dapi_cufftDoubleReal element, void *callerInfo, void *sharedPointer)
                {
                    return Z2DStoreMulMulIncreImpl((dapi_cufftDoubleReal *)dataOut, offset, element, (Z2DStoreMulMulIncreInfo_t)callerInfo);
                }
                extern __device__ dapi_cufftCallbackStoreD Z2DStoreMulMulIncreDevicePtr;

                ////////////////////////////////////////////////////////////////////////////////
                // CallBack Function
                ////////////////////////////////////////////////////////////////////////////////
                struct Z2DStoreMulHalfMulIncreInfo_st
                {
                    const dapi_cufftDoubleReal *m;
                    const dapi_cufftDoubleReal *incre_mul_src;
                    dapi_cufftDoubleReal *incre_dst;
                };
                using Z2DStoreMulHalfMulIncreInfo_t = Z2DStoreMulHalfMulIncreInfo_st *;
                __device__ __forceinline__ void Z2DStoreMulHalfMulIncreImpl(dapi_cufftDoubleReal *dataOut, size_t offset, dapi_cufftDoubleReal element, Z2DStoreMulHalfMulIncreInfo_t data, void *sharedPointer = nullptr)
                {
                    auto val = element * data->m[offset];
                    dataOut[offset] = val;
                    data->incre_dst[offset] += 0.5 * data->incre_mul_src[offset] * val;
                }
                __device__ __forceinline__ void Z2DStoreMulHalfMulIncre(void *dataOut, size_t offset, dapi_cufftDoubleReal element, void *callerInfo, void *sharedPointer)
                {
                    return Z2DStoreMulHalfMulIncreImpl((dapi_cufftDoubleReal *)dataOut, offset, element, (Z2DStoreMulHalfMulIncreInfo_t)callerInfo);
                }
                extern __device__ dapi_cufftCallbackStoreD Z2DStoreMulHalfMulIncreDevicePtr;

            }

            namespace detail
            {

                /// @brief a helper class to store the callback information for cufft.
                ///        Note that direct use of this class is ***ALMOST NEVER*** intended, since this class has ***NO DEVICE CONTROL***.
                ///        It must be manually guranteed that any creation of the instance of this class must happen after the specification of the deivce by native CUDA API or equivalents.
                /// @tparam DataT the type of the callback information
                template <typename DataT>
                class CBStorage
                {
                public:
                    enum ValueFields
                    {
                        DEVICE_POINTER = 0,
                        DEVICE_CONTAINER_INDEX = 1
                    };
                    // using DataT = std::tuple<double const *, double const *, double *, double>;

                    using RawDataT = uint64_t;
                    constexpr static std::size_t slice_size = (sizeof(DataT) + sizeof(RawDataT) - 1) / sizeof(RawDataT);

                    using HostRawDataT = std::array<uint64_t, slice_size>;
                    using DeviceRawDataT = RawDataT;
                    using ValueT = std::tuple<void *, std::size_t>;
                    using UnorderedMapT = boost::unordered_map<HostRawDataT, ValueT>;
                    using HostVectorT = std::vector<DataT>;
                    using DeviceVectorT = thrust::device_vector<DeviceRawDataT>;

                    auto register_data(const DataT &data) -> void *;
                    auto find(const DataT &data) const -> typename UnorderedMapT::const_iterator;
                    auto cend() const -> typename UnorderedMapT::const_iterator;
                    auto get_data_device_pointer(const DataT &data) const -> void *;
                    auto get_data_index(const DataT &data) const -> std::size_t;
                    auto clear() -> void;

                private:
                    HostVectorT h_vec_;
                    UnorderedMapT map_;
                    DeviceVectorT d_vec_;

                    static auto make_raw_data(const DataT &data) -> HostRawDataT;

                    void *old_d_vec_start_ = nullptr;
                };

                template <typename DataT>
                auto CBStorage<DataT>::register_data(const DataT &data) -> void *
                {
                    auto raw_data = make_raw_data(data);
                    auto itr = map_.find(raw_data);

                    if (itr != map_.end())
                    {
                        // current device Data has already been recorded:
                        // return the recorded data pointer.
                        return std::get<ValueFields::DEVICE_POINTER>(itr->second);
                    }
                    else
                    {
                        // current device Data is not recorded:

                        // add the record on host vector
                        h_vec_.push_back(data);
                        // add the corresponding record on device vector
                        for (std::size_t itr_slice = 0; itr_slice < slice_size; ++itr_slice)
                        {
                            d_vec_.push_back(raw_data[itr_slice]);
                        }
                        // record on the map
                        std::size_t current_index = h_vec_.size() - 1;
                        auto current_leading_ptr = thrust::raw_pointer_cast(&(d_vec_[current_index * slice_size]));
                        map_.emplace(raw_data, ValueT{current_leading_ptr, current_index});

                        auto d_vec_start = (void *)(thrust::raw_pointer_cast(&(d_vec_[0])));
                        if (old_d_vec_start_ != d_vec_start)
                        {
                            old_d_vec_start_ = d_vec_start;
                            // reallocation detected, fix all the pointers stored
                            for (auto &ele : map_)
                            {
                                std::get<ValueFields::DEVICE_POINTER>(ele.second) = ((RawDataT *)d_vec_start + std::get<ValueFields::DEVICE_CONTAINER_INDEX>(ele.second) * slice_size);
                            }
                        }

                        return current_leading_ptr;
                    }
                }

                template <typename DataT>
                auto CBStorage<DataT>::find(const DataT &data) const -> typename CBStorage<DataT>::UnorderedMapT::const_iterator
                {
                    return map_.find(make_raw_data(data));
                }

                template <typename DataT>
                auto CBStorage<DataT>::cend() const -> typename CBStorage<DataT>::UnorderedMapT::const_iterator
                {
                    return map_.cend();
                }

                template <typename DataT>
                auto CBStorage<DataT>::get_data_device_pointer(const DataT &data) const -> void *
                {
                    auto itr = find_data(data);
                    if (itr == map_.cend())
                    {
                        qutility::message::exit_with_message("CBStorage Error: the input data has not been registered. Consider to register it.", __FILE__, __LINE__);
                    }
                    else
                    {
                        return std::get<ValueFields::DEVICE_POINTER>(itr->second);
                    }
                }
                template <typename DataT>
                auto CBStorage<DataT>::get_data_index(const DataT &data) const -> std::size_t
                {
                    auto itr = find_data(data);
                    if (itr == map_.cend())
                    {
                        qutility::message::exit_with_message("CBStorage Error: the input data has not been registered. Consider to register it.", __FILE__, __LINE__);
                    }
                    else
                    {
                        return std::get<ValueFields::DEVICE_CONTAINER_INDEX>(itr->second);
                    }
                }

                template <typename DataT>
                auto CBStorage<DataT>::clear() -> void
                {
                    h_vec_.clear();
                    d_vec_.clear();
                    map_.clear();
                }

                template <typename DataT>
                auto CBStorage<DataT>::make_raw_data(const DataT &data) -> HostRawDataT
                {
                    alignas(DataT) RawDataT slice_buffer[slice_size];
                    auto slice_ptr = new (slice_buffer) DataT{data};
                    HostRawDataT ans;
                    for (std::size_t itr_slice = 0; itr_slice < slice_size; ++itr_slice)
                    {
                        ans[itr_slice] = slice_buffer[itr_slice];
                    }
                    return ans;
                }
            }

        }
    }
}