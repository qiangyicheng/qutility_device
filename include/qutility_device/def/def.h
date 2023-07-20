#pragma once

#include "config.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QUTILITY_DEVICE_FORCE_INLINE __forceinline__
#define QUTILITY_DEVICE_DEVICE_ACCESSABLE __device__
#define QUTILITY_DEVICE_HOST_ACCESSABLE __host__
#define QUTILITY_DEVICE_BOTH_ACCESSABLE __host__ __device__
#else
#define QUTILITY_DEVICE_FORCE_INLINE inline
#define QUTILITY_DEVICE_DEVICE_ACCESSABLE
#define QUTILITY_DEVICE_HOST_ACCESSABLE
#define QUTILITY_DEVICE_BOTH_ACCESSABLE
#endif