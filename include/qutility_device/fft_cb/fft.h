#pragma once

#include <optional>

#include "qutility/box.h"

#include "qutility_device/workspace.h"

#include "device_api/device_api_cufft.h"

namespace qutility
{
    namespace device
    {
        namespace fft_cb
        {
            namespace detail
            {
                auto create_plan() -> dapi_cufftHandle;
                auto destroy_plan(dapi_cufftHandle plan) -> void;
                auto get_workspace_size_requirement(dapi_cufftHandle plan) -> std::size_t;
                auto check_workspace_size(dapi_cufftHandle plan, qutility::device::workspace::Workspace<double> &working) -> bool;
                auto make_forward_plan_impl(dapi_cufftHandle forward, const qutility::box::Box<3> &box, std::size_t n_batch) -> void;
                auto make_forward_plan(dapi_cufftHandle forward, const qutility::box::Box<3> &box, std::size_t n_batch) -> void;
                auto make_forward_plan(dapi_cufftHandle forward, const qutility::box::Box<3> &box, std::size_t n_batch, qutility::device::workspace::Workspace<double> &working) -> void;
                auto make_backward_plan_impl(dapi_cufftHandle backward, const qutility::box::Box<3> &box, std::size_t n_batch) -> void;
                auto make_backward_plan(dapi_cufftHandle backward, const qutility::box::Box<3> &box, std::size_t n_batch) -> void;
                auto make_backward_plan(dapi_cufftHandle backward, const qutility::box::Box<3> &box, std::size_t n_batch, qutility::device::workspace::Workspace<double> &working) -> void;

                class FFTPlan
                {
                public:
                    enum class DIRECTION
                    {
                        FORWARD = 0,
                        BACKWARD = 1
                    };
                    using BoxT = qutility::box::Box<3>;

                    FFTPlan() = delete;
                    FFTPlan(const FFTPlan &) = delete;
                    FFTPlan &operator=(const FFTPlan &) = delete;
                    FFTPlan(FFTPlan &&) = delete;
                    FFTPlan &operator=(FFTPlan &&) = delete;

                    FFTPlan(DIRECTION dir, const BoxT &box, std::size_t n_batch);
                    FFTPlan(DIRECTION dir, const BoxT &box, std::size_t n_batch, const std::shared_ptr<qutility::device::workspace::Workspace<double>> &working);
                    FFTPlan(DIRECTION dir, const BoxT &box, std::size_t n_batch, dapi_cudaStream_t stream);
                    FFTPlan(DIRECTION dir, const BoxT &box, std::size_t n_batch, std::optional<dapi_cudaStream_t> stream, const std::shared_ptr<qutility::device::workspace::Workspace<double>> &working);
                    ~FFTPlan();

                    const DIRECTION dir_;
                    const BoxT box_;
                    const std::size_t n_batch_;
                    const dapi_cufftHandle plan_;
                    const std::shared_ptr<qutility::device::workspace::Workspace<double>> working_;

                private:
                };

            }
        }
    }
}