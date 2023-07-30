#include "qutility_device/fft_cb/fft.h"

#include <string_view>

#include "qutility/message.h"

#include "device_api/device_api_helper.h"

namespace qutility
{
    namespace device
    {
        namespace fft_cb
        {
            namespace detail
            {
                using namespace std::string_literals;
                /// @brief create a fft plan
                auto create_plan() -> dapi_cufftHandle
                {
                    dapi_cufftHandle plan;
                    dapi_checkCudaErrors(dapi_cufftCreate(&plan));
                    return plan;
                }

                /// @brief destroy a fft plan
                auto destroy_plan(dapi_cufftHandle plan) -> void
                {
                    dapi_checkCudaErrors(dapi_cufftDestroy(plan));
                }

                /// @brief get the requirement of the workspace size of a fft_plan
                auto get_workspace_size_requirement(dapi_cufftHandle plan) -> std::size_t
                {
                    std::size_t buffer_size;
                    dapi_checkCudaErrors(dapi_cufftGetSize(plan, &buffer_size));
                    return buffer_size;
                }

                /// @brief check whether a workspace has enough space for a fft plan
                auto check_workspace_size(dapi_cufftHandle plan, qutility::device::workspace::Workspace<double> &working) -> bool
                {
                    return working.size_in_bytes() >= get_workspace_size_requirement(plan);
                }

                auto make_forward_plan_impl(dapi_cufftHandle forward, const qutility::box::Box<3> &box, std::size_t n_batch) -> void
                {
                    constexpr int dim = 3;
                    long long int n[dim];
                    long long int embed_real[dim];
                    long long int embed_complex[dim];
                    const std::size_t dim_shift = dim - box.useful_dim_;
                    for (int itr = 0; itr < dim; ++itr)
                    {
                        n[itr] = box.compressed_box_.box_size_[itr];
                        embed_real[itr] = box.compressed_box_.box_size_[itr];
                        embed_complex[itr] = box.compressed_box_.box_size_hermit_[itr];
                    }

                    size_t buffer_size;
                    dapi_checkCudaErrors(dapi_cufftMakePlanMany64(
                        forward,
                        box.useful_dim_, n + dim_shift,
                        embed_real + dim_shift, 1, box.compressed_box_.total_size_,
                        embed_complex + dim_shift, 1, box.compressed_box_.total_size_hermit_,
                        CUFFT_D2Z, n_batch, &buffer_size));
                }
                auto make_forward_plan(dapi_cufftHandle forward, const qutility::box::Box<3> &box, std::size_t n_batch) -> void
                {
                    dapi_checkCudaErrors(dapi_cufftSetAutoAllocation(forward, true));
                    make_forward_plan_impl(forward, box, n_batch);
                }
                auto make_forward_plan(dapi_cufftHandle forward, const qutility::box::Box<3> &box, std::size_t n_batch, qutility::device::workspace::Workspace<double> &working) -> void
                {
                    dapi_checkCudaErrors(dapi_cufftSetAutoAllocation(forward, false));
                    make_forward_plan_impl(forward, box, n_batch);
                    if (!check_workspace_size(forward, working))
                    {
                        qutility::message::exit_with_message("make_forward_plan Error: Not enough space in the workspace for the fft calculation. The minimal requirement is "s + std::to_string(get_workspace_size_requirement(forward)) + ", while the provided workspace has the size of "s + std::to_string(working.size_in_bytes()), __FILE__, __LINE__);
                    }
                    dapi_checkCudaErrors(dapi_cufftSetWorkArea(forward, working.pointer()));
                }
                auto make_backward_plan_impl(dapi_cufftHandle backward, const qutility::box::Box<3> &box, std::size_t n_batch) -> void
                {
                    constexpr int dim = 3;
                    long long int n[dim];
                    long long int embed_real[dim];
                    long long int embed_complex[dim];
                    const std::size_t dim_shift = dim - box.useful_dim_;
                    for (int itr = 0; itr < dim; ++itr)
                    {
                        n[itr] = box.compressed_box_.box_size_[itr];
                        embed_real[itr] = box.compressed_box_.box_size_[itr];
                        embed_complex[itr] = box.compressed_box_.box_size_hermit_[itr];
                    }

                    size_t buffer_size;
                    dapi_checkCudaErrors(dapi_cufftMakePlanMany64(
                        backward,
                        box.useful_dim_, n + dim_shift,
                        embed_complex + dim_shift, 1, box.compressed_box_.total_size_hermit_,
                        embed_real + dim_shift, 1, box.compressed_box_.total_size_,
                        CUFFT_Z2D, n_batch, &buffer_size));
                }
                auto make_backward_plan(dapi_cufftHandle backward, const qutility::box::Box<3> &box, std::size_t n_batch) -> void
                {
                    dapi_checkCudaErrors(dapi_cufftSetAutoAllocation(backward, true));
                    make_backward_plan_impl(backward, box, n_batch);
                }
                auto make_backward_plan(dapi_cufftHandle backward, const qutility::box::Box<3> &box, std::size_t n_batch, qutility::device::workspace::Workspace<double> &working) -> void
                {
                    dapi_checkCudaErrors(dapi_cufftSetAutoAllocation(backward, false));
                    make_backward_plan_impl(backward, box, n_batch);
                    if (!check_workspace_size(backward, working))
                    {
                        qutility::message::exit_with_message("make_backward_plan Error: Not enough space in the workspace for the fft calculation. The minimal requirement is "s + std::to_string(get_workspace_size_requirement(backward)) + ", while the provided workspace has the size of "s + std::to_string(working.size_in_bytes()), __FILE__, __LINE__);
                    }
                    dapi_checkCudaErrors(dapi_cufftSetWorkArea(backward, working.pointer()));
                }

                /// @brief create an fft plan without specifying device and stream, and let the library allocate the workspace
                ///        Note that no device check is performed.
                FFTPlan::FFTPlan(DIRECTION dir, const BoxT &box, std::size_t n_batch)
                    : FFTPlan(dir, box, n_batch, std::nullopt, nullptr)
                {
                }

                /// @brief create an fft plan without specifying device and stream, and use the given workspace
                ///        Note that no device check is performed.
                FFTPlan::FFTPlan(DIRECTION dir, const BoxT &box, std::size_t n_batch, const std::shared_ptr<qutility::device::workspace::Workspace<double>> &working)
                    : FFTPlan(dir, box, n_batch, std::nullopt, working)
                {
                }

                /// @brief create an fft plan on the specified stream, and let the library allocate the workspace
                ///        Note that no device check is performed.
                FFTPlan::FFTPlan(DIRECTION dir, const BoxT &box, std::size_t n_batch, dapi_cudaStream_t stream)
                    : FFTPlan(dir, box, n_batch, stream, nullptr)
                {
                }

                /// @brief create an fft plan on the specified stream, and use the given workspace
                ///        Note that if stream is not specified, fft plan will not been associate to any stream, and its behavior will follow that of the library.
                ///        If the ptr to workspace is a nullptr, allocation will be handled by the library
                ///        Note that no device check is performed.
                FFTPlan::FFTPlan(DIRECTION dir, const BoxT &box, std::size_t n_batch, std::optional<dapi_cudaStream_t> stream, const std::shared_ptr<qutility::device::workspace::Workspace<double>> &working)
                    : dir_(dir), box_(box), n_batch_(n_batch), plan_(create_plan()), working_(working)
                {
                    if (working_)
                    {
                        switch (dir_)
                        {
                        case DIRECTION::FORWARD:
                            make_forward_plan(plan_, box_, n_batch_, *working_);
                            break;
                        case DIRECTION::BACKWARD:
                            make_backward_plan(plan_, box_, n_batch_, *working_);
                            break;
                        default:
                            // std::unreachable() //C++23
                            break;
                        }
                    }
                    else
                    {
                        switch (dir_)
                        {
                        case DIRECTION::FORWARD:
                            make_forward_plan(plan_, box_, n_batch_);
                            break;
                        case DIRECTION::BACKWARD:
                            make_backward_plan(plan_, box_, n_batch_);
                            break;
                        default:
                            // std::unreachable() //C++23
                            break;
                        }
                    }
                    if (stream)
                    {
                        dapi_checkCudaErrors(dapi_cufftSetStream(plan_, stream.value()));
                    }
                }

                FFTPlan::~FFTPlan()
                {
                    dapi_checkCudaErrors(dapi_cufftSetStream(plan_, nullptr));
                    destroy_plan(plan_);
                }
            }
        }
    }
}
