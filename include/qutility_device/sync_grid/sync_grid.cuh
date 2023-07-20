#pragma once

#include "device_api/device_api_cuda_device.h"

#ifdef QUTILITY_DEVICE_USE_ATOMIC_GRID_SYNC
#else
#include "device_api/device_api_cooperative_groups.h"
#endif

#include "qutility_device/def.h"

namespace qutility
{
    namespace device
    {
        namespace sync_grid
        {
            namespace detail
            {
                static QUTILITY_DEVICE_FORCE_INLINE QUTILITY_DEVICE_DEVICE_ACCESSIBLE unsigned int atomic_add(volatile unsigned int *addr, unsigned int val)
                {
                    unsigned int old;
                    old = dapi_atomicAdd((unsigned int *)addr, val);
                    return old;
                }

                static QUTILITY_DEVICE_FORCE_INLINE QUTILITY_DEVICE_DEVICE_ACCESSIBLE bool bar_has_flipped(unsigned int old_arrive, unsigned int current_arrive)
                {
                    // printf("check bar, arrived = %X, oldArrive = %X, from block %u \n", current_arrive, old_arrive, blockIdx.x);
                    return (((old_arrive ^ current_arrive) & 0x80000000) != 0);
                }

                static QUTILITY_DEVICE_FORCE_INLINE QUTILITY_DEVICE_DEVICE_ACCESSIBLE void bar_flush(volatile unsigned int *addr)
                {
                    dapi___threadfence();
                }

                template <bool is_first>
                static QUTILITY_DEVICE_FORCE_INLINE QUTILITY_DEVICE_DEVICE_ACCESSIBLE void sync_grids(unsigned int expected, volatile unsigned int *arrived)
                {
                    bool cta_master = (threadIdx.x + threadIdx.y + threadIdx.z == 0);
                    bool gpu_master = (blockIdx.x + blockIdx.y + blockIdx.z == 0);

                    dapi___syncthreads();

                    if (cta_master)
                    {
                        if constexpr (is_first)
                        {
                            unsigned int temp = *arrived;
                            dapi___threadfence();
                            if (temp != 0)
                            {
                                if (gpu_master)
                                    *arrived = 0;
                            }
                            if (temp != 0)
                                while (*arrived)
                                    ;
                            dapi___threadfence();
                        }

                        unsigned int nb = 1;
                        if (gpu_master)
                        {
                            nb = 0x80000000 - (expected - 1);
                        }

                        unsigned int oldArrive;
                        // printf("before atomic add,  temp = %X, nb = %X, arrived = %X, from block %u \n", temp, nb, *arrived, blockIdx.x);
                        oldArrive = atomic_add(arrived, nb);

                        while (!bar_has_flipped(oldArrive, *arrived))
                            ;

                        // flush barrier upon leaving
                        bar_flush((unsigned int *)arrived);
                    }

                    dapi___syncthreads();
                }
            }
            template <bool is_first>
            QUTILITY_DEVICE_FORCE_INLINE QUTILITY_DEVICE_DEVICE_ACCESSIBLE void sync(unsigned int *bar)
            {
                unsigned int expected = gridDim.x * gridDim.y * gridDim.z;

                detail::sync_grids<is_first>(expected, bar);
            }
        }
    }
}

#ifdef QUTILITY_DEVICE_USE_ATOMIC_GRID_SYNC
#define QUTILITY_DEVICE_SYNC_GRID_PREPARE
#define QUTILITY_DEVICE_SYNC_GRID_SYNC_FIRST(working) qutility::device::sync_grid::sync<true>((working - QUTILITY_DEVICE_WORKING_PREPEND_SIZE / (sizeof(decltype(*working)))));
#define QUTILITY_DEVICE_SYNC_GRID_SYNC_REST(working) qutility::device::sync_grid::sync<false>((working - QUTILITY_DEVICE_WORKING_PREPEND_SIZE / (sizeof(decltype(*working)))));
#define QUTILITY_DEVICE_SYNC_GRID_SYNC(working) QUTILITY_DEVICE_SYNC_GRID_SYNC_REST(working)
#else
#define QUTILITY_DEVICE_SYNC_GRID_PREPARE auto grid = cooperative_groups::this_grid();
#define QUTILITY_DEVICE_SYNC_GRID_SYNC(working) grid.sync();
#endif // QUTILITY_DEVICE_USE_ATOMIC_GRID_SYNC
