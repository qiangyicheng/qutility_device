// c++ headers
#include <iostream>

// gtest headers
#include <gtest/gtest.h>

// project headers
#include <qutility_device/info.h>

// other headers
#include <qutility/array_wrapper/array_wrapper_gpu.h>

TEST(QutilityDeviceInfoTest, MaxBlocksAndThreads)
{
    int device = 0;
    auto [max_blocks, max_threads] = qutility::device::info::device_max_blocks_and_threads(device);
    EXPECT_GE(max_blocks, 0);
    EXPECT_GE(max_threads, 0);

    qutility::array_wrapper::ArrayGPU<double> arr(0., 64, device);
    EXPECT_EQ(qutility::device::info::ptr_device_id(arr.pointer()), device);

    auto min_threads = qutility::device::info::device_min_nice_threads(device);
    EXPECT_GE(max_threads, min_threads);
}
