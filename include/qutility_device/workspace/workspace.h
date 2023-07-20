#pragma once

#include <cstddef>
#include <memory>

#include "qutility_device/event.h"
#include "qutility_device/def.h"

namespace qutility
{
    namespace device
    {
        namespace workspace
        {
            class WorkspaceRaw
            {
            public:
                WorkspaceRaw() = delete;
                WorkspaceRaw(const WorkspaceRaw &) = delete;
                WorkspaceRaw &operator=(const WorkspaceRaw &) = delete;
                WorkspaceRaw(WorkspaceRaw &&) = delete;
                WorkspaceRaw &operator=(WorkspaceRaw &&) = delete;

                explicit WorkspaceRaw(std::size_t size, int device); // size in bytes
                ~WorkspaceRaw();

                auto size_in_bytes() const -> std::size_t;

            public:
                constexpr static size_t prefix_size_ = QUTILITY_DEVICE_WORKING_PREPEND_SIZE;

            private:
                struct DataImpl;

                std::unique_ptr<DataImpl> data_impl_ptr_;

                const std::size_t size_;

            protected:
                void *const pointer_;
            };

            template <typename ValT>
            class Workspace : public qutility::device::event::StreamEventHelper, public WorkspaceRaw
            {
            public:
                Workspace() = delete;
                Workspace(const Workspace &) = delete;
                Workspace &operator=(const Workspace &) = delete;
                Workspace(Workspace &&) = delete;
                Workspace &operator=(Workspace &&) = delete;

                Workspace(size_t size, int device) : StreamEventHelper(device), WorkspaceRaw(size * sizeof(ValT), device) {} // size in ValT
                ~Workspace() {}

                using device_pointer_t = ValT *;

                operator device_pointer_t() const { return (device_pointer_t)pointer_; }
                auto pointer() const -> device_pointer_t { return (device_pointer_t)pointer_; }
            };

        }
    }
}