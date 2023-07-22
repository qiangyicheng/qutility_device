#pragma once

#include <cstddef>
#include <memory>

#include "dual_field_declare.h"

#include "qutility/array_wrapper/array_wrapper_gpu.h"

#include "qutility_device/event.h"

namespace qutility
{
    namespace device
    {
        namespace dual_field
        {
            /// @brief Two fields of ValT on device
            /// @tparam ValT the type of the element of the field
            template <typename ValT>
            class DualField : public event::StreamEventHelper
            {
            public:
                using value_type = ValT;
                using array_type = qutility::array_wrapper::DArrayGPU<ValT>;

                DualField() = delete;
                DualField(const DualField &) = delete;
                DualField &operator=(const DualField &) = delete;
                DualField(DualField &&) = delete;
                DualField &operator=(DualField &&) = delete;
                ~DualField() = default;

                DualField(std::size_t size, int device)
                    : size_(size), StreamEventHelper(device), field_(ValT{}, size, device), field_diff_(ValT{}, size, device) {}

                const std::size_t size_;

                array_type field_;
                array_type field_diff_;
            };

            /// @brief Two fields of ValT on device, with corresponding fields on host
            /// @tparam ValT the type of the element of the field
            template <typename ValT>
            class DualFieldEx : public DualField<ValT>
            {
            public:
                DualFieldEx() = delete;
                DualFieldEx(const DualFieldEx &) = delete;
                DualFieldEx &operator=(const DualFieldEx &) = delete;
                DualFieldEx(DualFieldEx &&) = delete;
                DualFieldEx &operator=(DualFieldEx &&) = delete;
                ~DualFieldEx() = default;

                DualFieldEx(std::size_t size, int device)
                    : DualField<ValT>(size, device), field_host_(ValT{}, size), field_diff_host_(ValT{}, size) {}

                /// @brief sync field data from host to device. This function DOES NOT imply a synchronization.
                /// @return the event that marks the completion of the copy. Note that another successive call that reuse the event may change the meaning
                template <typename... dapi_cudaStreamOrEventType>
                auto host_to_device(dapi_cudaStreamOrEventType... dependencies) -> dapi_cudaEvent_t
                {
                    this->set_device();
                    this->this_wait_other(dependencies...);
                    qutility::array_wrapper::array_copy_async(this->field_, field_host_, 0, this->stream_);
                    qutility::array_wrapper::array_copy_async(this->field_diff_, field_diff_host_, 0, this->stream_);
                    return this->record_event();
                }

                /// @brief sync field data from device to host. This function DOES NOT imply a synchronization.
                /// @return the event that marks the completion of the copy. Note that another successive call that reuse the event may change the meaning
                template <typename... dapi_cudaStreamOrEventType>
                auto device_to_host(dapi_cudaStreamOrEventType... dependencies) -> dapi_cudaEvent_t
                {
                    this->set_device();
                    this->this_wait_other(dependencies...);
                    qutility::array_wrapper::array_copy(field_host_, this->field_, 0);
                    qutility::array_wrapper::array_copy(field_diff_host_, this->field_diff_, 0);
                    return this->record_event();
                }

                using host_array_type = qutility::array_wrapper::DArrayDDRPinned<ValT>;

                host_array_type field_host_;
                host_array_type field_diff_host_;
            };
        }
    }
}
