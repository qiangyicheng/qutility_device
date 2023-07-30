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
                extern __device__ cufftCallbackStoreZ D2ZStoreMulRealDevicePtr;

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
                    using HostVectorT = thrust::device_vector<DataT>;
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