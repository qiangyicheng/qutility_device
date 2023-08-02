// c++ headers
#include <iostream>
#include <cmath>

// gtest headers
#include <gtest/gtest.h>

// project headers
#include <qutility_device/math.cuh>
#include <qutility_device/field.h>

template <size_t Rank, bool IfLastDimModified = false, size_t ThreadsPerBlock = 256, size_t MaxBlocksPerDim = 16>
__global__ void calc_eigenvalue_ref(double *k, double *working, double factorx, double factory, double factorz, int Nx, int Ny, int Nz);

TEST(QutilityDeviceMath, SetValue)
{
    using ValT = double;
    constexpr int device = 0;
    constexpr std::size_t size = 1024 * 8;
    constexpr std::size_t dup = 4;
    constexpr ValT coef = 0.33;
    constexpr ValT index = 1.233;

    qutility::device::field::FieldEx<ValT> in(size, device);
    qutility::device::field::FieldEx<ValT> out(size * dup, device);

    for (std::size_t itr = 0; itr < size; ++itr)
    {
        in.field_host_[itr] = (double)rand() / RAND_MAX;
    }
    in.host_to_device();
    in.sync_device();

    in.launch_kernel<256>(qutility::device::math::array_exp_dup<double>, {size, dup, out.field_, in.field_, coef, index}, 0);
    in.sync_device();

    out.device_to_host();
    out.sync_device();

    for (std::size_t itr_dup = 0; itr_dup < dup; ++itr_dup)
    {
        for (std::size_t itr = 0; itr < size; ++itr)
        {
            EXPECT_DOUBLE_EQ(out.field_host_[itr_dup * size + itr], coef * std::exp(index * in.field_host_[itr]));
        }
    }
}

TEST(QutilityDeviceMath, LaplacianEigenvalue1D)
{
    using ValT = double;
    constexpr int device = 0;
    constexpr std::size_t Nx = 128;
    constexpr double lx = 2.3;
    // constexpr std::size_t size = Nx;
    constexpr std::size_t size_hermit = Nx / 2 + 1;

    qutility::device::field::FieldEx<ValT> k_new(size_hermit, device);
    qutility::device::field::FieldEx<ValT> k_old(size_hermit, device);

    k_new.launch_kernel(qutility::device::math::laplacian_eigenvalue_1D<>, {7, 1, 1}, {128, 1, 1}, {k_new.field_, 1. / (lx * lx), Nx}, 0);
    k_old.launch_kernel<256>(calc_eigenvalue_ref<1, false, 256, 4>, {k_old.field_, nullptr, 0, 0, 1. / (lx * lx), 0, 0, Nx}, 0);

    k_new.device_to_host();
    k_old.device_to_host();
    k_new.sync_device();
    k_old.sync_device();
    for (std::size_t itr = 0; itr < size_hermit; ++itr)
    {
        EXPECT_DOUBLE_EQ(k_new.field_host_[itr], k_old.field_host_[itr]);
    }
}

TEST(QutilityDeviceMath, LaplacianEigenvalue2D)
{
    using ValT = double;
    constexpr int device = 0;
    constexpr std::size_t Nx = 128;
    constexpr std::size_t Ny = 96;
    constexpr double lx = 2.3;
    constexpr double ly = 4.3;
    constexpr std::size_t size = Nx * Ny;
    constexpr std::size_t size_hermit = Nx * (Ny / 2 + 1);

    qutility::device::field::Field<ValT> working(size, device);
    qutility::device::field::FieldEx<ValT> k_new(size_hermit, device);
    qutility::device::field::FieldEx<ValT> k_old(size_hermit, device);

    k_new.launch_kernel_cg(qutility::device::math::laplacian_eigenvalue_2D<>, {7, 1, 1}, {127, 1, 1}, {k_new.field_, working.field_, 1. / (lx * lx), 1. / (ly * ly), Nx, Ny}, 0);
    k_old.launch_kernel_cg<256>(calc_eigenvalue_ref<2, false, 256, 4>, {k_old.field_, working.field_, 0, 1. / (lx * lx), 1. / (ly * ly), 0, Nx, Ny}, 0, k_new.stream_);

    k_new.device_to_host();
    k_old.device_to_host();
    k_new.sync_device();
    k_old.sync_device();
    for (std::size_t itr = 0; itr < size_hermit; ++itr)
    {
        EXPECT_DOUBLE_EQ(k_new.field_host_[itr], k_old.field_host_[itr]) << itr / ((Ny / 2 + 1)) << "," << itr % (Ny / 2 + 1);
    }
}

TEST(QutilityDeviceMath, LaplacianEigenvalue3D)
{
    using ValT = double;
    constexpr int device = 0;
    constexpr std::size_t Nx = 128;
    constexpr std::size_t Ny = 96;
    constexpr std::size_t Nz = 32;
    constexpr double lx = 2.3;
    constexpr double ly = 4.3;
    constexpr double lz = 11.3;
    constexpr std::size_t size = Nx * Ny * Nz;
    constexpr std::size_t size_hermit = Nx * Ny * (Nz / 2 + 1);

    qutility::device::field::Field<ValT> working(size, device);
    qutility::device::field::FieldEx<ValT> k_new(size_hermit, device);
    qutility::device::field::FieldEx<ValT> k_old(size_hermit, device);

    k_new.launch_kernel_cg(qutility::device::math::laplacian_eigenvalue_3D<>, {7, 1, 1}, {127, 1, 1}, {k_new.field_, working.field_, 1. / (lx * lx), 1. / (ly * ly), 1. / (lz * lz), Nx, Ny, Nz}, 0);
    k_old.launch_kernel_cg<256>(calc_eigenvalue_ref<3, false, 256, 4>, {k_old.field_, working.field_, 1. / (lx * lx), 1. / (ly * ly), 1. / (lz * lz), Nx, Ny, Nz}, 0, k_new.stream_);

    k_new.device_to_host();
    k_old.device_to_host();
    k_new.sync_device();
    k_old.sync_device();
    for (std::size_t itr = 0; itr < size_hermit; ++itr)
    {
        EXPECT_DOUBLE_EQ(k_new.field_host_[itr], k_old.field_host_[itr]) << itr / (Ny * (Nz / 2 + 1)) << "," << (itr / (Nz / 2 + 1)) % Ny << "," << itr / ((Nz / 2 + 1) * Ny);
    }
}

template <size_t Rank, bool IfLastDimModified, size_t ThreadsPerBlock, size_t MaxBlocksPerDim>
__global__ void calc_eigenvalue_ref(double *k, double *working, double factorx, double factory, double factorz, int Nx, int Ny, int Nz)
{
    using namespace qutility::device::math;
    constexpr static size_t threads_per_block = ThreadsPerBlock;
    QUTILITY_DEVICE_SYNC_GRID_PREPARE;

    static_assert(utility::next_pow_2(threads_per_block) == threads_per_block, "threads_per_block must be power of 2.");

    int thread_rank = blockIdx.x * blockDim.x + threadIdx.x;

    int Nz_complex = Nz / 2 + 1;

    if constexpr (Rank == 1)
    {
        int grid_size = gridDim.x * blockDim.x;
        // working, factorx and factory omitted
        for (auto itr_z = thread_rank; itr_z < Nz_complex; itr_z += grid_size)
        {
            auto val = itr_z * itr_z * factorz;
            if constexpr (IfLastDimModified)
                val *= (2. - ((itr_z % ((Nz + 1) / 2)) == 0));

            k[itr_z] = val;
        }
    }
    else if constexpr (Rank == 2)
    {
        constexpr static size_t max_blocks_per_dim_ = MaxBlocksPerDim;

        // blocks for eigenvals on y direction
        size_t required_block_for_y = (Ny + threads_per_block - 1) / threads_per_block;
        size_t block_for_y = required_block_for_y > max_blocks_per_dim_ ? max_blocks_per_dim_ : required_block_for_y;
        size_t thread_for_y = block_for_y * threads_per_block;
        size_t shift_for_y = 0;
        auto ky = working + shift_for_y;

        // blocks for eigenvals on z direction
        size_t required_block_for_z = (Nz_complex + threads_per_block - 1) / threads_per_block;
        size_t block_for_z = required_block_for_z > max_blocks_per_dim_ ? max_blocks_per_dim_ : required_block_for_z;
        size_t thread_for_z = block_for_z * threads_per_block;
        size_t shift_for_z = utility::next_pow_2(Ny);
        auto kz = working + shift_for_z;

        if (blockIdx.x < block_for_y)
        {
            for (size_t itr_y = thread_rank; itr_y < Ny; itr_y += thread_for_y)
            {
                ky[itr_y] = (double)itr_y - (int)(itr_y > ((Ny + 1) / 2)) * (double)Ny;
                ky[itr_y] *= ky[itr_y];
                ky[itr_y] *= factory;
            }
        }
        else if (blockIdx.x < block_for_y + block_for_z)
        {
            for (size_t itr_z = thread_rank - thread_for_y; itr_z < Nz_complex; itr_z += thread_for_z)
            {
                kz[itr_z] = itr_z * itr_z * factorz;
            }
        }

        QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));

        for (size_t itr_y = blockIdx.x; itr_y < Ny; itr_y += gridDim.x)
        {
            for (size_t itr_z = threadIdx.x; itr_z < Nz_complex; itr_z += blockDim.x)
            {
                auto val = ky[itr_y] + kz[itr_z];
                if constexpr (IfLastDimModified)
                    val *= (2. - ((itr_z % ((Nz + 1) / 2)) == 0));

                k[itr_y * Nz_complex + itr_z] = val;
            }
        }
    }
    else if constexpr (Rank == 3)
    {
        constexpr static size_t max_blocks_per_dim_ = MaxBlocksPerDim;

        // blocks for eigenvals on x direction
        size_t required_block_for_x = (Nx + threads_per_block - 1) / threads_per_block;
        size_t block_for_x = required_block_for_x > max_blocks_per_dim_ ? max_blocks_per_dim_ : required_block_for_x;
        size_t thread_for_x = block_for_x * threads_per_block;
        size_t shift_for_x = 0;
        auto kx = working + shift_for_x;

        // blocks for eigenvals on y direction
        size_t required_block_for_y = (Ny + threads_per_block - 1) / threads_per_block;
        size_t block_for_y = required_block_for_y > max_blocks_per_dim_ ? max_blocks_per_dim_ : required_block_for_y;
        size_t thread_for_y = block_for_y * threads_per_block;
        size_t shift_for_y = utility::next_pow_2(Nx);
        auto ky = working + shift_for_y;

        // blocks for eigenvals on z direction
        size_t required_block_for_z = (Nz_complex + threads_per_block - 1) / threads_per_block;
        size_t block_for_z = required_block_for_z > max_blocks_per_dim_ ? max_blocks_per_dim_ : required_block_for_z;
        size_t thread_for_z = block_for_z * threads_per_block;
        size_t shift_for_z = shift_for_y + utility::next_pow_2(Ny);
        auto kz = working + shift_for_z;

        if (blockIdx.x < block_for_x)
        {
            for (size_t itr_x = thread_rank; itr_x < Nx; itr_x += thread_for_x)
            {
                kx[itr_x] = (double)itr_x - (int)(itr_x > ((Nx + 1) / 2)) * (double)Nx;
                kx[itr_x] *= kx[itr_x];
                kx[itr_x] *= factorx;
            }
        }
        else if (blockIdx.x < block_for_x + block_for_y)
        {
            for (size_t itr_y = thread_rank - thread_for_x; itr_y < Ny; itr_y += thread_for_y)
            {
                ky[itr_y] = (double)itr_y - (int)(itr_y > ((Ny + 1) / 2)) * (double)Ny;
                ky[itr_y] *= ky[itr_y];
                ky[itr_y] *= factory;
            }
        }
        else if (blockIdx.x < block_for_x + block_for_y + block_for_z)
        {
            for (size_t itr_z = thread_rank - thread_for_x - thread_for_y; itr_z < Nz_complex; itr_z += thread_for_z)
            {
                kz[itr_z] = itr_z * itr_z * factorz;
            }
        }

        QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));

        for (size_t itr_block = blockIdx.x; itr_block < Nx * Ny; itr_block += gridDim.x)
        {
            __shared__ double valxy;
            if (threadIdx.x == 0)
                valxy = kx[itr_block / Ny] + ky[itr_block % Ny];
            dapi___syncthreads();
            for (size_t itr_z = threadIdx.x; itr_z < Nz_complex; itr_z += blockDim.x)
            {
                auto val = valxy + kz[itr_z];
                if constexpr (IfLastDimModified)
                    val *= (2. - ((itr_z % ((Nz + 1) / 2)) == 0));

                k[itr_block * Nz_complex + itr_z] = val;
            }
        }
    }
    else
        static_assert(Rank <= 3, "Dimension higher than 3 is not supported");
}
