#include "qutility/array_wrapper/array_wrapper_gpu.h"

#include "qutility_device/workspace.h"

namespace qutility
{
    namespace device
    {
        namespace workspace
        {
            struct WorkspaceRaw::DataImpl : public qutility::array_wrapper::DArrayGPU<uint8_t>
            {
                DataImpl(std::size_t size, int device)
                    : qutility::array_wrapper::DArrayGPU<uint8_t>(0 /*must be zero*/, size, device)
                {
                }
            };

            WorkspaceRaw::WorkspaceRaw( std::size_t size, int device)
                : size_(size),
                  data_impl_ptr_(std::make_unique<DataImpl>(size + prefix_size_, device)),
                  pointer_(data_impl_ptr_->pointer() + prefix_size_)
            {
            }

            WorkspaceRaw::~WorkspaceRaw()
            {
            }

            auto WorkspaceRaw::size_in_bytes() const -> std::size_t
            {
                return size_;
            }

        }
    }
}