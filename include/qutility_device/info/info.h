#pragma once

#include <cstddef>
#include <tuple>

namespace qutility
{
    namespace device
    {
        namespace info
        {
            /// @brief obtain the device ID of the device pointer. Obtain device ID of a host pointer is undefine
            /// @param ptr device pointer
            /// @return device id
            auto ptr_device_id(const void *ptr) -> int;

            /// @brief obtain the maximum number of blocks and threads that can run concurrently, namely, the upper limit of the cooperative groups.
            /// @param device device id
            /// @return [max_blocks, max_threads]
            auto device_max_blocks_and_threads(int device) -> std::pair<std::size_t, std::size_t>;

            /// @brief obtain the minimum number of threads that can fill into an SM. 
            /// @param device 
            /// @return number of threads
            auto device_min_nice_threads(int device) -> std::size_t;
        }
    }
}