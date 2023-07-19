// c++ headers
#include <iostream>

// gtest headers
#include <gtest/gtest.h>

// project headers
#include <qutility_device/event.h>

// other headers
#include <qutility/array_wrapper/array_wrapper_gpu.h>
#include <qutility/array_wrapper/array_wrapper_cpu.h>

template <std::size_t ThreadsPerBlock = 256>
__global__ void add(double *dst, const double *src1, const double *src2, int size)
{
    int thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;
    for (int itr = thread_rank; itr < size; itr += grid_size)
    {
        dst[itr] = src1[itr] + src2[itr];
    }
}

TEST(QutilityDeviceEvent, KernelLaunch)
{
    int device = 0;

    qutility::device::event::StreamEventHelper se;
    se.create_stream_and_event(device);
    se.sync_device();

    constexpr std::size_t test_size = 1024 * 8;
    constexpr std::size_t ThreadsPerBlock = 256;

    qutility::array_wrapper::ArrayGPU<double> src1(0., test_size, device);
    qutility::array_wrapper::ArrayGPU<double> src2(0., test_size, device);
    qutility::array_wrapper::ArrayGPU<double> dst(0., test_size, device);
    qutility::array_wrapper::ArrayDDRPinned<double> dst_host(0., test_size);
    qutility::array_wrapper::ArrayDDR<double> dst_ref(0., test_size);

    {
        // normal kernel launch
        for (std::size_t itr = 0; itr < test_size; ++itr)
        {
            auto randval1 = (double)rand() / RAND_MAX;
            auto randval2 = (double)rand() / RAND_MAX;
            src1[itr] = randval1;
            src2[itr] = randval2;
            dst_ref[itr] = randval1 + randval2;
        }
        se.launch_kernel(add<ThreadsPerBlock>, {1024, 1, 1}, {ThreadsPerBlock, 1, 1}, {dst.pointer(), src1.pointer(), src2.pointer(), test_size}, 0);
        qutility::array_wrapper::array_copy(dst_host, dst);
        se.sync_device();
        for (std::size_t itr = 0; itr < test_size; ++itr)
        {
            EXPECT_DOUBLE_EQ(dst_ref[itr], dst_host[itr]);
        }
    }

    {
        // normal kernel launch
        for (std::size_t itr = 0; itr < test_size; ++itr)
        {
            auto randval1 = (double)rand() / RAND_MAX;
            auto randval2 = (double)rand() / RAND_MAX;
            src1[itr] = randval1;
            src2[itr] = randval2;
            dst_ref[itr] = randval1 + randval2;
        }
        se.launch_kernel<ThreadsPerBlock>(add<ThreadsPerBlock>, {dst.pointer(), src1.pointer(), src2.pointer(), test_size}, 0);
        qutility::array_wrapper::array_copy(dst_host, dst);
        se.sync_device();
        for (std::size_t itr = 0; itr < test_size; ++itr)
        {
            EXPECT_DOUBLE_EQ(dst_ref[itr], dst_host[itr]);
        }
    }

    {
        // cg kernel launch
        for (std::size_t itr = 0; itr < test_size; ++itr)
        {
            auto randval1 = (double)rand() / RAND_MAX;
            auto randval2 = (double)rand() / RAND_MAX;
            src1[itr] = randval1;
            src2[itr] = randval2;
            dst_ref[itr] = randval1 + randval2;
        }
        se.launch_kernel_cg(add<ThreadsPerBlock>, {16, 1, 1}, {ThreadsPerBlock, 1, 1}, {dst.pointer(), src1.pointer(), src2.pointer(), test_size}, 0);
        qutility::array_wrapper::array_copy(dst_host, dst);
        se.sync_device();
        for (std::size_t itr = 0; itr < test_size; ++itr)
        {
            EXPECT_DOUBLE_EQ(dst_ref[itr], dst_host[itr]);
        }
    }

    {
        // cg kernel launch with default grid settings
        for (std::size_t itr = 0; itr < test_size; ++itr)
        {
            auto randval1 = (double)rand() / RAND_MAX;
            auto randval2 = (double)rand() / RAND_MAX;
            src1[itr] = randval1;
            src2[itr] = randval2;
            dst_ref[itr] = randval1 + randval2;
        }
        se.launch_kernel_cg<ThreadsPerBlock>(add<ThreadsPerBlock>, {dst.pointer(), src1.pointer(), src2.pointer(), test_size}, 0);
        qutility::array_wrapper::array_copy(dst_host, dst);
        se.sync_device();
        for (std::size_t itr = 0; itr < test_size; ++itr)
        {
            EXPECT_DOUBLE_EQ(dst_ref[itr], dst_host[itr]);
        }
    }
}