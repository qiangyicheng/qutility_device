#pragma once

#include <cstddef>
#include <memory>

#include "field_declare.h"

#include "qutility/array_wrapper/array_wrapper_gpu.h"

#include "qutility_device/event.h"

namespace qutility
{
    namespace device
    {
        namespace field
        {

            /// @brief Field of ValT on device
            /// @tparam ValT the type of the element of the field
            template <typename ValT>
            class Field : public event::StreamEventHelper
            {
            public:
                using value_type = ValT;
                using array_type = qutility::array_wrapper::DArrayGPU<ValT>;

                Field() = delete;
                Field(const Field &) = delete;
                Field &operator=(const Field &) = delete;
                Field(Field &&) = delete;
                Field &operator=(Field &&) = delete;
                ~Field() = default;

                Field(std::size_t size, int device)
                    : size_(size), StreamEventHelper(device), field_(ValT{}, size, device) {}

                ValT * pointer() { return field_.pointer(); }
                const ValT * pointer() const { return field_.pointer(); }

                const std::size_t size_;

                array_type field_;
            };

            /// @brief Field of ValT on device, with corresponding field on host
            /// @tparam ValT the type of the element of the field
            template <typename ValT>
            class FieldEx : public Field<ValT>
            {
            public:
                FieldEx() = delete;
                FieldEx(const FieldEx &) = delete;
                FieldEx &operator=(const FieldEx &) = delete;
                FieldEx(FieldEx &&) = delete;
                FieldEx &operator=(FieldEx &&) = delete;
                ~FieldEx() = default;

                FieldEx(std::size_t size, int device)
                    : Field<ValT>(size, device), field_host_(ValT{}, size) {}

                /// @brief sync field data from host to device. This function DOES NOT imply a synchronization.
                /// @return the event that marks the completion of the copy. Note that another successive call that reuse the event may change the meaning
                template <typename... dapi_cudaStreamOrEventType>
                auto host_to_device(dapi_cudaStreamOrEventType... dependencies) -> dapi_cudaEvent_t
                {
                    this->set_device();
                    this->this_wait_other(dependencies...);
                    qutility::array_wrapper::array_copy_async(this->field_, field_host_, 0, this->stream_);
                    return this->record_event();
                }

                /// @brief sync field data from device to host. This function DOES NOT imply a synchronization.
                /// @return the event that marks the completion of the copy. Note that another successive call that reuse the event may change the meaning
                template <typename... dapi_cudaStreamOrEventType>
                auto device_to_host(dapi_cudaStreamOrEventType... dependencies) -> dapi_cudaEvent_t
                {
                    this->set_device();
                    this->this_wait_other(dependencies...);
                    qutility::array_wrapper::array_copy_async(field_host_, this->field_, 0, this->stream_);
                    return this->record_event();
                }

                ValT * pointer_host() { return field_host_.pointer(); }
                const ValT *  pointer_host() const { return field_host_.pointer(); }

                using host_array_type = qutility::array_wrapper::DArrayDDRPinned<ValT>;

                host_array_type field_host_;
            };

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

                ValT * pointer() { return field_.pointer(); }
                const ValT *  pointer() const { return field_.pointer(); }
                ValT * pointer_diff() { return field_diff_.pointer(); }
                const ValT *  pointer_diff() const { return field_diff_.pointer(); }

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
                    qutility::array_wrapper::array_copy_async(field_host_, this->field_, 0, this->stream_);
                    qutility::array_wrapper::array_copy_async(field_diff_host_, this->field_diff_, 0, this->stream_);
                    return this->record_event();
                }

                ValT * pointer_host() { return field_host_.pointer(); }
                const ValT *  pointer_host() const { return field_host_.pointer(); }
                ValT * pointer_diff_host() { return field_diff_host_.pointer(); }
                const ValT *  pointer_diff_host() const { return field_diff_host_.pointer(); }

                using host_array_type = qutility::array_wrapper::DArrayDDRPinned<ValT>;

                host_array_type field_host_;
                host_array_type field_diff_host_;
            };
        }
    }
}
