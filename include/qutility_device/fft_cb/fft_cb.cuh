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
                        auto load_data_host = cb_load_data_.register_data(load_data);
                        dapi_checkCudaErrors(dapi_cufftXtClearCallback(this->plan_, call_back_load_t_));
                        dapi_checkCudaErrors(dapi_cufftXtSetCallback(this->plan_, (void **)&load_kernel_host_ptr_, call_back_load_t_, &load_data_host));
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
                        auto store_data_host = cb_store_data_.register_data(store_data);
                        dapi_checkCudaErrors(dapi_cufftXtClearCallback(this->plan_, call_back_store_t_));
                        dapi_checkCudaErrors(dapi_cufftXtSetCallback(this->plan_, (void **)&store_kernel_host_ptr_, call_back_store_t_, &store_data_host));
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
                        dapi_checkCudaErrors(dapi_cufftExecD2Z(this->plan_, input, output));
                    }
                    else
                    {
                        dapi_checkCudaErrors(dapi_cufftExecZ2D(this->plan_, input, output));
                    }
                    return record_event();
                }

                detail::CBStorage<CallbackLoadDataFallbackT> cb_load_data_;
                detail::CBStorage<CallbackStoreDataFallbackT> cb_store_data_;
                const CallbackLoadKernelT load_kernel_host_ptr_;
                const CallbackStoreKernelT store_kernel_host_ptr_;

            private:
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