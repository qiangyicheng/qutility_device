#pragma once

namespace qutility
{
    namespace device
    {
        namespace math
        {
            namespace utility
            {
                __device__ __forceinline__ constexpr auto next_pow_2(size_t x) -> size_t
                {
                    --x;
                    x |= x >> 1;
                    x |= x >> 2;
                    x |= x >> 4;
                    x |= x >> 8;
                    x |= x >> 16;
                    x |= x >> 32;
                    return ++x;
                }
            }
        }
    }
}