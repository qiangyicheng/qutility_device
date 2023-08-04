#pragma once

#include <cstddef>
#include <tuple>
#include <optional>
#include <utility>

#include "qutility/traits.h"
#include "qutility/tuple_utility.h"

#include "device_api/device_api_cuda_runtime.h"

#include "qutility_device/info.h"

namespace qutility
{
    namespace device
    {
        namespace event
        {
            /// @brief let a stream wait for the event(s)
            /// @param stream stream that will wait
            auto stream_wait_event(dapi_cudaStream_t stream) -> void;

            /// @brief let a stream wait for the event(s)
            /// @param stream stream that will wait
            /// @param First first event to wait for
            auto stream_wait_event(dapi_cudaStream_t stream, dapi_cudaEvent_t First) -> void;

            /// @brief let a stream wait for the event(s)
            /// @tparam ...dapi_cudaEventType must be dapi_cudaEvent_t
            /// @param stream stream that will wait
            /// @param First first event to wait for
            /// @param ...Rest other event to wait for
            template <typename... dapi_cudaEventType>
            inline auto stream_wait_event(dapi_cudaStream_t stream, dapi_cudaEvent_t First, dapi_cudaEventType... Rest) -> void
            {
                static_assert(qutility::traits::is_list<dapi_cudaEvent_t, dapi_cudaEventType...>::value, "All dapi_cudaEventType must be dapi_cudaEvent_t.");
                stream_wait_event(stream, First);
                stream_wait_event(stream, Rest...);
                return;
            }

            class StreamEventHelper
            {
            public:
                StreamEventHelper(int device);
                StreamEventHelper();
                ~StreamEventHelper();

                template <typename CallableKernelT>
                using func_args_tuple = qutility::traits::func_args_tuple<CallableKernelT>;

            public:
                auto set_device() const -> void;
                auto sync_device() const -> void;
                auto create_stream_and_event(int device) -> void;
                auto destroy_stream_and_event() -> void;
                auto record_event() const -> dapi_cudaEvent_t;

                auto check_pointer() const -> void;
                auto check_pointer(const void *ptr) const -> void;
                template <typename DataTFirst, typename... DataTRest>
                auto check_pointer(const DataTFirst *first, const DataTRest *...rest) const -> void;

                auto check_number_of_threads(std::size_t num) const -> void;
                auto check_number_of_threads(dim3 dim_block) const -> void;

                auto check_number_of_blocks_cg(std::size_t num) const -> void;
                auto check_number_of_blocks_cg(dim3 dim_grid) const -> void;

                auto this_wait_other() const -> void;
                template <typename dapi_cudaStreamOrEventFirstType, typename... dapi_cudaStreamOrEventType>
                auto this_wait_other(dapi_cudaStreamOrEventFirstType First, dapi_cudaStreamOrEventType... Rest) const -> void;

                auto other_wait_this() const -> void;
                template <typename dapi_cudaStreamFirstType, typename... dapi_cudaStreamType>
                auto other_wait_this(dapi_cudaStreamFirstType First, dapi_cudaStreamType... Rest) const -> void;

                template <typename CallableKernelT, typename... dapi_cudaStreamOrEventType>
                auto launch_kernel(CallableKernelT kernel, dim3 dim_grid, dim3 dim_block, func_args_tuple<CallableKernelT> paras, size_t sharedMem, dapi_cudaStreamOrEventType... dependencies) const -> dapi_cudaEvent_t;
                template <std::size_t ThreadsPerBlock, typename CallableKernelT, typename... dapi_cudaStreamOrEventType>
                auto launch_kernel(CallableKernelT kernel, func_args_tuple<CallableKernelT> paras, size_t sharedMem, dapi_cudaStreamOrEventType... dependencies) const -> dapi_cudaEvent_t;

                template <typename CallableKernelT, typename... dapi_cudaStreamOrEventType>
                auto launch_kernel_cg(CallableKernelT kernel, dim3 dim_grid, dim3 dim_block, func_args_tuple<CallableKernelT> paras, size_t sharedMem, dapi_cudaStreamOrEventType... dependencies) const -> dapi_cudaEvent_t;
                template <std::size_t ThreadsPerBlock, typename CallableKernelT, typename... dapi_cudaStreamOrEventType>
                auto launch_kernel_cg(CallableKernelT kernel, func_args_tuple<CallableKernelT> paras, size_t sharedMem, dapi_cudaStreamOrEventType... dependencies) const -> dapi_cudaEvent_t;

            public:
                dapi_cudaStream_t stream_ = nullptr;
                dapi_cudaEvent_t finish_ = nullptr;
                dapi_cudaEvent_t to_wait_for_ = nullptr;

                /// @brief max number of blocks for cooperative group
                std::size_t max_blocks_cg_ = 0;
                /// @brief max number of threads per block
                std::size_t max_threads_per_block_ = 0;

            protected:
                std::optional<int> device_id_;

                auto other_wait_this_impl(dapi_cudaStream_t other) const -> void;
                auto this_wait_other_impl(dapi_cudaStream_t other) const -> void;
                auto this_wait_other_impl(dapi_cudaEvent_t other) const -> void;
                auto launch_kernel_impl(void *func, dim3 dim_grid, dim3 dim_block, void **arg_ptr_table, std::size_t sharedMem) const -> void;
                auto launch_kernel_cg_impl(void *func, dim3 dim_grid, dim3 dim_block, void **arg_ptr_table, std::size_t sharedMem) const -> void;
            };

            /// @brief Check whether the pointers belong to the device of current helper.
            template <typename DataTFirst, typename... DataTRest>
            auto StreamEventHelper::check_pointer(const DataTFirst *first, const DataTRest *...rest) const -> void
            {
                check_pointer((const void *)first);
                check_pointer(rest...);
            }

            /// @brief Require that all the listed streams will wait for the completetion of the last event in current stream.
            ///        Note that this does not mean a synchronization
            template <typename dapi_cudaStreamFirstType, typename... dapi_cudaStreamType>
            auto StreamEventHelper::other_wait_this(dapi_cudaStreamFirstType First, dapi_cudaStreamType... Rest) const -> void
            {
                other_wait_this_impl(First);
                other_wait_this(Rest...);
                return;
            }

            /// @brief Require that current stream will wait for the completion of the listed events or streams.
            ///        Note that this does not mean a synchronization
            template <typename dapi_cudaStreamOrEventFirstType, typename... dapi_cudaStreamOrEventType>
            auto StreamEventHelper::this_wait_other(dapi_cudaStreamOrEventFirstType First, dapi_cudaStreamOrEventType... Rest) const -> void
            {
                this_wait_other_impl(First);
                this_wait_other(Rest...);
                return;
            }

            /// @brief Launch normal kernel on the device specified in this helper class.
            ///        The input parameters follow the convention of cudaLaunchKernel, while the parameters can be filled by a initializer list of a automatically generated tuple, similar to the <<< >>> compiler extension of nvcc.
            ///        The parameter pack expects the dependencies this kernel launch will depend on.
            ///        Note that this does not mean a synchronization.
            /// @param kernel kernel to launch. DO NOT convert it to void* in advance.
            /// @param dim_grid dimension of the grid
            /// @param dim_block dimension of the block
            /// @param paras all parameters of the kernel
            /// @param sharedMem size of the shared memory
            /// @param ...dependencies all events or streams to depend on
            /// @return the event that marks the finish of the kernel
            template <typename CallableKernelT, typename... dapi_cudaStreamOrEventType>
            auto StreamEventHelper::launch_kernel(CallableKernelT kernel, dim3 dim_grid, dim3 dim_block, func_args_tuple<CallableKernelT> paras, size_t sharedMem, dapi_cudaStreamOrEventType... dependencies) const -> dapi_cudaEvent_t
            {
                set_device();
                check_number_of_threads(dim_block);
                this_wait_other(dependencies...);
                auto paras_ptr = qutility::tuple_utility::make_tuple_ptrs(paras);
                launch_kernel_impl(
                    (void *)kernel,
                    dim_grid, dim_block, (void **)(paras_ptr.begin()), sharedMem);
                return record_event();
            }

            /// @brief Launch normal kernel with 1D grid and 1D block on the device specified in this helper class.
            ///        Grid size and block size specification are skipped, while the kernel parameters can be filled by a initializer list of a automatically generated tuple, similar to the <<< >>> compiler extension of nvcc.
            ///        The parameter pack expects the dependencies this kernel launch will depend on.
            ///        The grid size is determined to be maximum number of concurrent blocks, where the threads per block are specified by the template parameter.
            ///        Note that this does not mean a synchronization.
            /// @tparam ThreadsPerBlock numer of threads per block. This value will be checked against the max number of threads per block
            /// @param kernel kernel to launch. DO NOT convert it to void* in advance.
            /// @param dim_grid dimension of the grid
            /// @param dim_block dimension of the block
            /// @param paras all parameters of the kernel
            /// @param sharedMem size of the shared memory
            /// @param ...dependencies all events or streams to depend on
            /// @return the event that marks the finish of the kernel
            template <std::size_t ThreadsPerBlock, typename CallableKernelT, typename... dapi_cudaStreamOrEventType>
            auto StreamEventHelper::launch_kernel(CallableKernelT kernel, func_args_tuple<CallableKernelT> paras, size_t sharedMem, dapi_cudaStreamOrEventType... dependencies) const -> dapi_cudaEvent_t
            {
                set_device();
                check_number_of_threads(ThreadsPerBlock);
                this_wait_other(dependencies...);
                auto paras_ptr = qutility::tuple_utility::make_tuple_ptrs(paras);
                launch_kernel_impl(
                    (void *)kernel,
                    dim3{(decltype(std::declval<dim3>().x))max_blocks_cg_, 1, 1}, dim3{ThreadsPerBlock, 1, 1}, (void **)(paras_ptr.begin()), sharedMem);
                return record_event();
            }

            /// @brief Launch cooperative kernel on the device specified in this helper class.
            ///        The input parameters follow the convention of cudaLaunchKernel, while the parameters can be filled by a initializer list of a automatically generated tuple, similar to the <<< >>> compiler extension of nvcc.
            ///        The parameter pack expects the dependencies this kernel launch will depend on.
            ///        For cooperative groups, the number of threads per block and the number of block are both limited.
            ///        Note that this does not mean a synchronization.
            /// @param kernel kernel to launch. DO NOT convert it to void* in advance.
            /// @param dim_grid dimension of the grid
            /// @param dim_block dimension of the block
            /// @param paras all parameters of the kernel
            /// @param sharedMem size of the shared memory
            /// @param ...dependencies all events or streams to depend on
            /// @return the event that marks the finish of the kernel
            template <typename CallableKernelT, typename... dapi_cudaStreamOrEventType>
            auto StreamEventHelper::launch_kernel_cg(CallableKernelT kernel, dim3 dim_grid, dim3 dim_block, func_args_tuple<CallableKernelT> paras, size_t sharedMem, dapi_cudaStreamOrEventType... dependencies) const -> dapi_cudaEvent_t
            {
                set_device();
                check_number_of_threads(dim_block);
                check_number_of_blocks_cg(dim_grid);
                this_wait_other(dependencies...);
                auto paras_ptr = qutility::tuple_utility::make_tuple_ptrs(paras);
                launch_kernel_cg_impl(
                    (void *)kernel,
                    dim_grid, dim_block, (void **)(paras_ptr.begin()), sharedMem);
                return record_event();
            }

            /// @brief Launch normal kernel with 1D grid and 1D block on the device specified in this helper class.
            ///        Grid size and block size specification are skipped, while the kernel parameters can be filled by a initializer list of a automatically generated tuple, similar to the <<< >>> compiler extension of nvcc.
            ///        The parameter pack expects the dependencies this kernel launch will depend on.
            ///        The grid size is determined to be maximum number of concurrent blocks, where the threads per block are specified by the template parameter.
            ///        Note that this does not mean a synchronization.
            /// @tparam ThreadsPerBlock numer of threads per block. This value will be checked against the max number of threads per block. CAUTION this value should be consistent with the kernel itself, which cannot be checked by this function.
            /// @param kernel kernel to launch. DO NOT convert it to void* in advance.
            /// @param dim_grid dimension of the grid
            /// @param dim_block dimension of the block
            /// @param paras all parameters of the kernel
            /// @param sharedMem size of the shared memory
            /// @param ...dependencies all events or streams to depend on
            /// @return the event that marks the finish of the kernel
            template <std::size_t ThreadsPerBlock, typename CallableKernelT, typename... dapi_cudaStreamOrEventType>
            auto StreamEventHelper::launch_kernel_cg(CallableKernelT kernel, func_args_tuple<CallableKernelT> paras, size_t sharedMem, dapi_cudaStreamOrEventType... dependencies) const -> dapi_cudaEvent_t
            {
                set_device();
                check_number_of_threads(ThreadsPerBlock);
                this_wait_other(dependencies...);
                auto paras_ptr = qutility::tuple_utility::make_tuple_ptrs(paras);
                launch_kernel_cg_impl(
                    (void *)kernel,
                    dim3{(decltype(std::declval<dim3>().z))max_blocks_cg_, 1, 1}, dim3{ThreadsPerBlock, 1, 1}, (void **)(paras_ptr.begin()), sharedMem);
                return record_event();
            }
        }
    }
}