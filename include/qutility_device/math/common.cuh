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

                template <int Order, typename ValT>
                __device__ __forceinline__ ValT fast_exponent(ValT val)
                {
                    if constexpr (Order == 0)
                    {
                        return 1.;
                    }
                    else if constexpr (Order == 1)
                    {
                        return val;
                    }
                    else if constexpr (Order == 2)
                    {
                        return val * val;
                    }
                    else
                    {
                        return fast_exponent<Order / 2>(val) * fast_exponent<Order - (Order / 2)>(val);
                    }
                }

            }
        }
    }
}