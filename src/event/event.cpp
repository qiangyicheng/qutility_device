#include <string>

#include "qutility_device/event.h"

#include "qutility_device/info.h"

#include "qutility/message.h"

#include "device_api/device_api_helper.h"
#include "device_api/device_api_cooperative_groups.h"

namespace qutility
{
    namespace device
    {
        namespace event
        {
            using namespace std::literals;

            auto stream_wait_event(dapi_cudaStream_t stream) -> void
            {
            }

            auto stream_wait_event(dapi_cudaStream_t stream, dapi_cudaEvent_t First) -> void
            {
                dapi_checkCudaErrors(dapi_cudaStreamWaitEvent(stream, First, 0));
            }

            StreamEventHelper::StreamEventHelper()
            {
            }

            StreamEventHelper::StreamEventHelper(int device)
            {
                create_stream_and_event(device);
            }

            StreamEventHelper::~StreamEventHelper()
            {
                destroy_stream_and_event();
            }

            /// @brief Set the device according to the value stored in device_id_
            auto StreamEventHelper::set_device() const -> void
            {
                if (device_id_)
                {
                    dapi_checkCudaErrors(dapi_cudaSetDevice(device_id_.value()));
                }
                else
                {
                    qutility::message::exit_with_message("StreamEventHelper Error: Device ID is not set. Do you forget to set it?", __FILE__, __LINE__);
                }
            }

            /// @brief synchronize device
            auto StreamEventHelper::sync_device() const -> void
            {
                set_device();
                dapi_checkCudaErrors(dapi_cudaDeviceSynchronize());
            }

            /// @brief Create stream and event on device. Old stream and event might be destroyed if the one requested is different from the existed ones.
            /// @param device device id
            auto StreamEventHelper::create_stream_and_event(int device) -> void
            {
                if (device < 0)
                {
                    qutility::message::exit_with_message("StreamEventHelper Error: Choose device with invalid device ID"s + std::to_string(device), __FILE__, __LINE__);
                }

                if (device_id_)
                {
                    if (device_id_.value() == device)
                    {
                        // set the same device ID, do nothing.
                        return;
                    }
                    else
                    {
                        destroy_stream_and_event();
                    }
                }
                device_id_ = device;
                dapi_checkCudaErrors(dapi_cudaSetDevice(device_id_.value()));
                dapi_checkCudaErrors(dapi_cudaStreamCreate(&stream_));
                dapi_checkCudaErrors(dapi_cudaEventCreate(&finish_));
                dapi_checkCudaErrors(dapi_cudaEventCreate(&to_wait_for_));
                auto [max_blocks, max_threads] = info::device_max_blocks_and_threads(device_id_.value());
                max_blocks_cg_ = max_blocks;
                max_threads_per_block_ = max_threads;
            }

            /// @brief Destroy stream and event on device
            auto StreamEventHelper::destroy_stream_and_event() -> void
            {
                if (device_id_)
                {
                    dapi_checkCudaErrors(dapi_cudaSetDevice(device_id_.value()));
                    dapi_checkCudaErrors(dapi_cudaStreamDestroy(stream_));
                    dapi_checkCudaErrors(dapi_cudaEventDestroy(finish_));
                    dapi_checkCudaErrors(dapi_cudaEventDestroy(to_wait_for_));
                    device_id_.reset();
                    max_blocks_cg_ = 0;
                    max_threads_per_block_ = 0;
                }
            }

            /// @brief Record event on current stream. Note that set_device() will be called.
            /// @return Return the returned event.
            auto StreamEventHelper::record_event() const -> dapi_cudaEvent_t
            {
                set_device();
                dapi_checkCudaErrors(dapi_cudaEventRecord(finish_, stream_));
                return finish_;
            }

            /// @brief Check whether the pointers belong to the device of current helper.
            auto StreamEventHelper::check_pointer() const -> void
            {
            }

            /// @brief Check whether the pointers belong to the device of current helper.
            auto StreamEventHelper::check_pointer(const void *ptr) const -> void
            {
                set_device();
                if (info::ptr_device_id(ptr) != device_id_.value())
                {
                    qutility::message::exit_with_message("StreamEventHelper Error: The device pointer check failed. Pointer "s + std::to_string((intptr_t)ptr) + " does not belong to device "s + std::to_string(device_id_.value()), __FILE__, __LINE__);
                }
            }

            /// @brief check whether the given number of threads exceed the limit. Note that this function is innocent of register and shared memory usage.
            auto StreamEventHelper::check_number_of_threads(std::size_t num) const -> void
            {
                set_device();
                if (num > max_threads_per_block_)
                {
                    qutility::message::exit_with_message("StreamEventHelper Error: The requested number of threads "s + std::to_string(num) + " exceed the limit of "s + std::to_string(max_threads_per_block_) + " on device "s + std::to_string(device_id_.value()), __FILE__, __LINE__);
                }
            }

            /// @brief check whether the given number of threads exceed the limit. Note that this function is innocent of register and shared memory usage.
            auto StreamEventHelper::check_number_of_threads(dim3 dim_block) const -> void
            {
                check_number_of_threads(dim_block.x * dim_block.y * dim_block.z);
            }

            /// @brief check whether the given number of blocks exceed the limit of the cooperative group.
            auto StreamEventHelper::check_number_of_blocks_cg(std::size_t num) const -> void
            {
                set_device();
                if (num > max_blocks_cg_)
                {
                    qutility::message::exit_with_message("StreamEventHelper Error: The requested number of blocks "s + std::to_string(num) + " exceed the limit of "s + std::to_string(max_blocks_cg_) + " for the cooperative group on device "s + std::to_string(device_id_.value()), __FILE__, __LINE__);
                }
            }

            auto StreamEventHelper::check_number_of_blocks_cg(dim3 dim_grid) const -> void
            {
                set_device();
                if (dim_grid.y != 1 || dim_grid.z != 1)
                {
                    qutility::message::exit_with_message("StreamEventHelper Error: The requested dimension of grid is not a 1D grid. Only 1D grid is allowed for cooperative group execution."s, __FILE__, __LINE__);
                }
                check_number_of_blocks_cg(dim_grid.x);
            }

            /// @brief Require that current stream will wait for the completion of the listed events or streams.
            ///        Note that this does not mean a synchronization
            auto StreamEventHelper::this_wait_other() const -> void
            {
            }

            /// @brief Require that all the listed streams will wait for the completetion of the last event in current stream.
            ///        Note that this does not mean a synchronization
            auto StreamEventHelper::other_wait_this() const -> void
            {
            }

            /// @brief Check whether the pointers belong to the device of current helper.
            auto StreamEventHelper::other_wait_this_impl(dapi_cudaStream_t other) const -> void
            {
                set_device();
                dapi_checkCudaErrors(dapi_cudaEventRecord(finish_, stream_));
                stream_wait_event(other, finish_);
            }

            auto StreamEventHelper::this_wait_other_impl(dapi_cudaStream_t other) const -> void
            {
                set_device();
                dapi_checkCudaErrors(dapi_cudaEventRecord(to_wait_for_, other));
                stream_wait_event(stream_, to_wait_for_);
            }

            auto StreamEventHelper::this_wait_other_impl(dapi_cudaEvent_t other) const -> void
            {
                set_device();
                stream_wait_event(stream_, other);
            }

            auto StreamEventHelper::launch_kernel_impl(void *func, dim3 dim_grid, dim3 dim_block, void **arg_ptr_table, std::size_t sharedMem) const -> void
            {
                set_device();
                dapi_checkCudaErrors(dapi_cudaLaunchKernel(func, dim_grid, dim_block, arg_ptr_table, sharedMem, stream_));
            }
            auto StreamEventHelper::launch_kernel_cg_impl(void *func, dim3 dim_grid, dim3 dim_block, void **arg_ptr_table, std::size_t sharedMem) const -> void
            {
                set_device();
                dapi_checkCudaErrors(dapi_cudaLaunchCooperativeKernel(func, dim_grid, dim_block, arg_ptr_table, sharedMem, stream_));
            }

        }
    }
}