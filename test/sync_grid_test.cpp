// c++ headers
#include <iostream>

// gtest headers
#include <gtest/gtest.h>

// project headers
#include <qutility_device/event.h>
#include <qutility_device/sync_grid.cuh>

// other headers
#include <qutility/array_wrapper/array_wrapper_gpu.h>
#include <qutility/array_wrapper/array_wrapper_cpu.h>

#include <device_api/device_api_cub.h>

template <std::size_t ThreadsPerBlock = 256>
__global__ void sum(double *dst, const double *src, int size, double *working)
{
    QUTILITY_DEVICE_SYNC_GRID_PREPARE;
    int thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;
    typedef dapi_cub::BlockReduce<double, ThreadsPerBlock, dapi_cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduceT;
    {
        __shared__ typename BlockReduceT::TempStorage temp_storage;
        double data = 0.0;
        for (int itr = thread_rank; itr < size; itr += grid_size)
            data += src[itr];
        dapi___syncthreads();
        double agg = BlockReduceT(temp_storage).Sum(data);
        if (threadIdx.x == 0)
            working[blockIdx.x] = agg;
        QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
        if (thread_rank == 0)
        {
            for (int itr_block = 1; itr_block < gridDim.x; ++itr_block)
                agg += working[itr_block];
            *dst = agg;
        }
        QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
    }
}

TEST(QutilityDeviceSyncGrid, Sum)
{
    int device = 0;

    qutility::device::event::StreamEventHelper se;
    se.create_stream_and_event(device);
    se.sync_device();

    constexpr std::size_t test_size = 1024 * 8;
    constexpr std::size_t ThreadsPerBlock = 256;

    qutility::array_wrapper::ArrayGPU<double> src(0., test_size, device);
    qutility::array_wrapper::ArrayGPU<double> working(0., test_size, device);
    qutility::array_wrapper::ArrayGPU<double> dst(0., 1, device);

    double ans_ref = 0;

    {
        // normal kernel launch
        ans_ref = 0;
        for (std::size_t itr = 0; itr < test_size; ++itr)
        {
            auto randval1 = (double)rand() / RAND_MAX;
            src[itr] = randval1;
            ans_ref += randval1;
        }
        se.launch_kernel_cg<ThreadsPerBlock>(sum<ThreadsPerBlock>, {dst.pointer(), src.pointer(), test_size, working.pointer()}, 0);
        se.sync_device();
        EXPECT_NEAR(dst[0], ans_ref, 1e-8);
    }

}