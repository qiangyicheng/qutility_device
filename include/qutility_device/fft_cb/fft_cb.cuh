#pragma once

#include <type_traits>

#include "fft.h"
#include "cb.cuh"

#include "qutility_device/event.h"
#include "qutility_device/workspace.h"

#include "device_api/device_api_cuda_runtime.h"
#include "device_api/device_api_cufft.h"

#include "device_api/device_api_helper.h"

namespace qutility
{
    namespace device
    {
        namespace fft_cb
        {
            namespace workaround
            {
                inline __global__ void run_cufftCallbackLoadD(dapi_cufftCallbackLoadD func, size_t size, void *dataIn, void *callerInfo, void *sharedPointer)
                {
                    size_t thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                    size_t grid_size = gridDim.x * blockDim.x;
                    for (size_t itr = thread_rank; itr < size; itr += grid_size)
                    {
                        ((dapi_cufftDoubleReal *)dataIn)[itr] = func(dataIn, itr, callerInfo, sharedPointer);
                    }
                }

                inline __global__ void run_cufftCallbackLoadZ(dapi_cufftCallbackLoadZ func, size_t size, void *dataIn, void *callerInfo, void *sharedPointer)
                {
                    size_t thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                    size_t grid_size = gridDim.x * blockDim.x;
                    for (size_t itr = thread_rank; itr < size; itr += grid_size)
                    {
                        ((dapi_cufftDoubleComplex *)dataIn)[itr] = func(dataIn, itr, callerInfo, sharedPointer);
                    }
                }

                inline __global__ void run_cufftCallbackStoreD(dapi_cufftCallbackStoreD func, size_t size, void *dataOut, void *callerInfo, void *sharedPointer)
                {
                    size_t thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                    size_t grid_size = gridDim.x * blockDim.x;
                    for (size_t itr = thread_rank; itr < size; itr += grid_size)
                    {
                        func(dataOut, itr, ((dapi_cufftDoubleReal *)dataOut)[itr], callerInfo, sharedPointer);
                    }
                }

                inline __global__ void run_cufftCallbackStoreZ(dapi_cufftCallbackStoreZ func, size_t size, void *dataOut, void *callerInfo, void *sharedPointer)
                {
                    size_t thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                    size_t grid_size = gridDim.x * blockDim.x;
                    for (size_t itr = thread_rank; itr < size; itr += grid_size)
                    {
                        func(dataOut, itr, ((dapi_cufftDoubleComplex *)dataOut)[itr], callerInfo, sharedPointer);
                    }
                }
            }

            using DIRECTION = detail::FFTPlan::DIRECTION;

            template <DIRECTION Dir, typename CBLoadDataT = void, typename CBStoreDataT = void>
            class FFTPlanWithCB
                : public event::StreamEventHelper,
                  public detail::FFTPlan
            {
            public:
                FFTPlanWithCB() = delete;
                FFTPlanWithCB(const FFTPlanWithCB &) = delete;
                FFTPlanWithCB &operator=(const FFTPlanWithCB &) = delete;
                FFTPlanWithCB(FFTPlanWithCB &&) = delete;
                FFTPlanWithCB &operator=(FFTPlanWithCB &&) = delete;
                ~FFTPlanWithCB() = default;

                constexpr static DIRECTION dir_ = Dir;
                using CallbackLoadDataT = CBLoadDataT;
                using CallbackStoreDataT = CBStoreDataT;

                constexpr static bool if_using_load_callback_ = !std::is_same_v<CallbackLoadDataT, void>;
                constexpr static bool if_using_store_callback_ = !std::is_same_v<CallbackStoreDataT, void>;
                using CallbackLoadDataFallbackT = typename qutility::traits::static_if<if_using_load_callback_, CallbackLoadDataT, double *>::type;
                using CallbackStoreDataFallbackT = typename qutility::traits::static_if<if_using_store_callback_, CallbackStoreDataT, double *>::type;

                using PlanInputDataT = typename qutility::traits::static_switch<static_cast<std::size_t>(dir_), cufftDoubleReal, cufftDoubleComplex>::type;
                using PlanOutputDataT = typename qutility::traits::static_switch<static_cast<std::size_t>(dir_), cufftDoubleComplex, cufftDoubleReal>::type;
                using CallbackLoadKernelT = typename qutility::traits::static_switch<static_cast<std::size_t>(dir_), dapi_cufftCallbackLoadD, dapi_cufftCallbackLoadZ>::type;
                using CallbackStoreKernelT = typename qutility::traits::static_switch<static_cast<std::size_t>(dir_), dapi_cufftCallbackStoreZ, dapi_cufftCallbackStoreD>::type;

                constexpr static dapi_cufftXtCallbackType call_back_load_t_ = dir_ == DIRECTION::FORWARD ? DAPI_CUFFT_CB_LD_REAL_DOUBLE : DAPI_CUFFT_CB_LD_COMPLEX_DOUBLE;

                constexpr static dapi_cufftXtCallbackType call_back_store_t_ = dir_ == DIRECTION::FORWARD ? DAPI_CUFFT_CB_ST_COMPLEX_DOUBLE : DAPI_CUFFT_CB_ST_REAL_DOUBLE;

                FFTPlanWithCB(int device, const BoxT &box, std::size_t n_batch, const CallbackLoadKernelT &load_kernel, const CallbackStoreKernelT &store_kernel, const std::shared_ptr<qutility::device::workspace::Workspace<double>> &working)
                    : StreamEventHelper(device),
                      FFTPlan(dir_, box, n_batch, this->stream_, working),
                      load_kernel_host_ptr_(get_callback_host_ptr(load_kernel)),
                      store_kernel_host_ptr_(get_callback_host_ptr(store_kernel))
                {
                }

                auto set_callback_load_data(CallbackLoadDataFallbackT load_data) -> void
                {
                    if constexpr (if_using_load_callback_)
                    {
                        this->set_device();
                        load_data_host_ = cb_load_data_.register_data(load_data);
#ifndef QUTILITY_DEVICE_CUFFT_CB_WORKAROUND_LD
                        dapi_checkCudaErrors(dapi_cufftXtClearCallback(this->plan_, call_back_load_t_));
                        dapi_checkCudaErrors(dapi_cufftXtSetCallback(this->plan_, (void **)&load_kernel_host_ptr_, call_back_load_t_, &load_data_host_));
#endif
                    }
                    else
                    {
                        qutility::message::exit_with_message("FFTPlanWithCB Error: You are setting callback data for a plan that the callback function is not configured.", __FILE__, __LINE__);
                    }
                }

                auto set_callback_store_data(CallbackStoreDataFallbackT store_data) -> void
                {
                    if constexpr (if_using_store_callback_)
                    {
                        this->set_device();
                        store_data_host_ = cb_store_data_.register_data(store_data);
#ifndef QUTILITY_DEVICE_CUFFT_CB_WORKAROUND_ST
                        dapi_checkCudaErrors(dapi_cufftXtClearCallback(this->plan_, call_back_store_t_));
                        dapi_checkCudaErrors(dapi_cufftXtSetCallback(this->plan_, (void **)&store_kernel_host_ptr_, call_back_store_t_, &store_data_host_));
#endif
                    }
                    else
                    {
                        qutility::message::exit_with_message("FFTPlanWithCB Error: You are setting callback data for a plan that the callback function is not configured.", __FILE__, __LINE__);
                    }
                }

                template <typename... dapi_cudaStreamOrEventType>
                auto execute(PlanInputDataT *input, PlanOutputDataT *output, dapi_cudaStreamOrEventType... dependencies) -> dapi_cudaEvent_t
                {
                    this->set_device();
                    this->this_wait_other(this->working_->stream_);
                    this->this_wait_other(dependencies...);
                    if constexpr (dir_ == DIRECTION::FORWARD)
                    {
#ifdef QUTILITY_DEVICE_CUFFT_CB_WORKAROUND_LD
                        if constexpr (if_using_load_callback_)
                            this->launch_kernel<256>(
                                workaround::run_cufftCallbackLoadD,
                                {load_kernel_host_ptr_, this->box_.compressed_box_.total_size_ * this->n_batch_, (void *)input, load_data_host_, nullptr},
                                0);
#endif
                        dapi_checkCudaErrors(dapi_cufftExecD2Z(this->plan_, input, output));
#ifdef QUTILITY_DEVICE_CUFFT_CB_WORKAROUND_ST
                        if constexpr (if_using_store_callback_)
                            this->launch_kernel<256>(
                                workaround::run_cufftCallbackStoreZ,
                                {store_kernel_host_ptr_, this->box_.compressed_box_.total_size_hermit_ * this->n_batch_, (void *)output, store_data_host_, nullptr},
                                0);
#endif
                    }
                    else
                    {
#ifdef QUTILITY_DEVICE_CUFFT_CB_WORKAROUND_LD
                        if constexpr (if_using_load_callback_)
                            this->launch_kernel<256>(
                                workaround::run_cufftCallbackLoadZ,
                                {load_kernel_host_ptr_, this->box_.compressed_box_.total_size_hermit_ * this->n_batch_, (void *)input, load_data_host_, nullptr},
                                0);
#endif
                        dapi_checkCudaErrors(dapi_cufftExecZ2D(this->plan_, input, output));
#ifdef QUTILITY_DEVICE_CUFFT_CB_WORKAROUND_ST
                        if constexpr (if_using_store_callback_)
                            this->launch_kernel<256>(
                                workaround::run_cufftCallbackStoreD,
                                {store_kernel_host_ptr_, this->box_.compressed_box_.total_size_ * this->n_batch_, (void *)output, store_data_host_, nullptr},
                                0);
#endif
                    }
                    this->other_wait_this(this->working_->stream_);
                    return record_event();
                }

                detail::CBStorage<CallbackLoadDataFallbackT> cb_load_data_;
                detail::CBStorage<CallbackStoreDataFallbackT> cb_store_data_;
                const CallbackLoadKernelT load_kernel_host_ptr_;
                const CallbackStoreKernelT store_kernel_host_ptr_;

            private:
                void *load_data_host_ = nullptr;
                void *store_data_host_ = nullptr;
                auto get_callback_host_ptr(const CallbackLoadKernelT &load) -> CallbackLoadKernelT
                {
                    if (if_using_load_callback_)
                    {
                        this->set_device();
                        CallbackLoadKernelT temp;
                        dapi_checkCudaErrors(dapi_cudaMemcpyFromSymbol(&temp, load, sizeof(CallbackLoadKernelT)));
                        return temp;
                    }
                    else
                    {
                        return nullptr;
                    }
                }
                auto get_callback_host_ptr(const CallbackStoreKernelT &store) -> CallbackStoreKernelT
                {
                    if (if_using_store_callback_)
                    {
                        this->set_device();
                        CallbackStoreKernelT temp;
                        dapi_checkCudaErrors(dapi_cudaMemcpyFromSymbol(&temp, store, sizeof(CallbackStoreKernelT)));
                        return temp;
                    }
                    else
                    {
                        return nullptr;
                    }
                }
            };
        }
    }
}