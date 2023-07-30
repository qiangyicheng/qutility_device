// c++ headers
#include <iostream>

// gtest headers
#include <gtest/gtest.h>

// project headers
#include <qutility_device/fft_cb.cuh>

// other headers
#include <qutility/array_wrapper/array_wrapper_cpu.h>
#include <qutility/array_wrapper/array_wrapper_gpu.h>

#include <fftw3.h>

TEST(QutilityDeviceFFTCallback, ForwardVoidVoid)
{
    using DIRECTION = qutility::device::fft_cb::DIRECTION;
    constexpr int device = 0;
    constexpr std::size_t Nx = 64;
    constexpr std::size_t Ny = 64;
    constexpr std::size_t Nz = 128;
    constexpr qutility::box::Box<3> box{{Nx, Ny, Nz}};
    constexpr std::size_t batch = 1;

    auto working = std::make_shared<qutility::device::workspace::Workspace<double>>(box.original_box_.total_size_ * 16, device);

    qutility::device::fft_cb::FFTPlanWithCB<DIRECTION::FORWARD, void, void> plan{
        device,
        box,
        batch,
        nullptr,
        nullptr,
        working};

    qutility::array_wrapper::ArrayGPU<double> in(1, box.original_box_.total_size_, device);
    qutility::array_wrapper::ArrayGPU<double> out(0, box.original_box_.total_size_hermit_ * 2, device);

    qutility::array_wrapper::ArrayDDRPinned<double> in_host(1, box.original_box_.total_size_);
    qutility::array_wrapper::ArrayDDRPinned<double> out_host(0, box.original_box_.total_size_hermit_ * 2);

    qutility::array_wrapper::ArrayDDR<double> out_ref(0, box.original_box_.total_size_hermit_ * 2);

    for (std::size_t itr = 0; itr < box.original_box_.total_size_; ++itr)
    {
        in_host[itr] = (double)rand() / RAND_MAX;
    }
    qutility::array_wrapper::array_copy(in, in_host);
    plan.execute(in.pointer(), (dapi_cufftDoubleComplex *)(out.pointer()));
    qutility::array_wrapper::array_copy(out_host, out);

    fftw_plan host_plan = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, in_host.pointer(), (fftw_complex *)(out_ref.pointer()), FFTW_ESTIMATE);
    fftw_execute(host_plan);

    for (std::size_t itr = 0; itr < box.original_box_.total_size_hermit_ * 2; ++itr)
    {
        EXPECT_NEAR(out_host[itr], out_ref[itr], std::sqrt(box.original_box_.total_size_) * 1e-13) << itr;
    }
}

TEST(QutilityDeviceFFTCallback, BackwardVoidVoid)
{
    using DIRECTION = qutility::device::fft_cb::DIRECTION;
    constexpr int device = 0;
    constexpr std::size_t Nx = 64;
    constexpr std::size_t Ny = 64;
    constexpr std::size_t Nz = 128;
    constexpr qutility::box::Box<3> box{{Nx, Ny, Nz}};
    constexpr std::size_t batch = 1;

    auto working = std::make_shared<qutility::device::workspace::Workspace<double>>(box.original_box_.total_size_ * 16, device);

    qutility::device::fft_cb::FFTPlanWithCB<DIRECTION::BACKWARD, void, void> plan{
        device,
        box,
        batch,
        nullptr,
        nullptr,
        working};

    qutility::array_wrapper::ArrayGPU<double> in(0, box.original_box_.total_size_hermit_ * 2, device);
    qutility::array_wrapper::ArrayGPU<double> out(1, box.original_box_.total_size_, device);

    qutility::array_wrapper::ArrayDDRPinned<double> in_host(0, box.original_box_.total_size_hermit_ * 2);
    qutility::array_wrapper::ArrayDDRPinned<double> out_host(1, box.original_box_.total_size_);

    qutility::array_wrapper::ArrayDDR<double> out_ref(0, box.original_box_.total_size_);

    for (std::size_t itr = 0; itr < box.original_box_.total_size_hermit_ * 2; ++itr)
    {
        in_host[itr] = (double)rand() / RAND_MAX;
    }
    qutility::array_wrapper::array_copy(in, in_host);
    plan.execute((dapi_cufftDoubleComplex *)(in.pointer()), out.pointer());
    qutility::array_wrapper::array_copy(out_host, out);

    fftw_plan host_plan = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, (fftw_complex *)(in_host.pointer()), out_ref.pointer(), FFTW_ESTIMATE);
    fftw_execute(host_plan);

    for (std::size_t itr = 0; itr < box.original_box_.total_size_; ++itr)
    {
        EXPECT_NEAR(out_host[itr], out_ref[itr], std::sqrt(box.original_box_.total_size_) * 1e-13) << itr;
    }
}

TEST(QutilityDeviceFFTCallback, ForwardMulMul)
{
    using DIRECTION = qutility::device::fft_cb::DIRECTION;
    constexpr int device = 0;
    constexpr std::size_t Nx = 64;
    constexpr std::size_t Ny = 64;
    constexpr std::size_t Nz = 128;
    constexpr qutility::box::Box<3> box{{Nx, Ny, Nz}};
    constexpr std::size_t batch = 1;

    auto working = std::make_shared<qutility::device::workspace::Workspace<double>>(box.original_box_.total_size_ * 16, device);

    qutility::device::fft_cb::FFTPlanWithCB<
        DIRECTION::FORWARD,
        qutility::device::fft_cb::kernel::D2ZLoadMulInfo_st,
        qutility::device::fft_cb::kernel::D2ZStoreMulRealInfo_st>
        plan{
            device,
            box,
            batch,
            qutility::device::fft_cb::kernel::D2ZLoadMulDevicePtr,
            qutility::device::fft_cb::kernel::D2ZStoreMulRealDevicePtr,
            working};

    qutility::array_wrapper::ArrayGPU<double> in(0, box.original_box_.total_size_, device);
    qutility::array_wrapper::ArrayGPU<double> mul_in(0, box.original_box_.total_size_, device);
    qutility::array_wrapper::ArrayGPU<double> out(0, box.original_box_.total_size_hermit_ * 2, device);
    qutility::array_wrapper::ArrayGPU<double> mul_out(0, box.original_box_.total_size_hermit_, device);

    qutility::array_wrapper::ArrayDDRPinned<double> in_host(1, box.original_box_.total_size_);
    qutility::array_wrapper::ArrayDDRPinned<double> mul_in_host(1, box.original_box_.total_size_);
    qutility::array_wrapper::ArrayDDRPinned<double> out_host(0, box.original_box_.total_size_hermit_ * 2);
    qutility::array_wrapper::ArrayDDRPinned<double> mul_out_host(0, box.original_box_.total_size_hermit_);

    qutility::array_wrapper::ArrayDDR<double> out_ref(0, box.original_box_.total_size_hermit_ * 2);

    for (std::size_t itr = 0; itr < box.original_box_.total_size_; ++itr)
    {
        in_host[itr] = (double)rand() / RAND_MAX;
        mul_in_host[itr] = (double)rand() / RAND_MAX;
    }
    for (std::size_t itr = 0; itr < box.original_box_.total_size_hermit_; ++itr)
    {
        mul_out_host[itr] = (double)rand() / RAND_MAX;
    }
    qutility::array_wrapper::array_copy(in, in_host);
    qutility::array_wrapper::array_copy(mul_in, mul_in_host);
    qutility::array_wrapper::array_copy(mul_out, mul_out_host);

    plan.set_callback_load_data({mul_in.pointer()});
    plan.set_callback_store_data({mul_out.pointer()});
    plan.execute(in.pointer(), (dapi_cufftDoubleComplex *)(out.pointer()));

    qutility::array_wrapper::array_copy(out_host, out);

    for (std::size_t itr = 0; itr < box.original_box_.total_size_; ++itr)
    {
        in_host[itr] *= mul_in_host[itr];
    }

    fftw_plan host_plan = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, in_host.pointer(), (fftw_complex *)(out_ref.pointer()), FFTW_ESTIMATE);
    fftw_execute(host_plan);

    for (std::size_t itr = 0; itr < box.original_box_.total_size_hermit_ * 2; ++itr)
    {
        out_ref[itr] *= mul_out_host[itr / 2];
    }

    for (std::size_t itr = 0; itr < box.original_box_.total_size_hermit_ * 2; ++itr)
    {
        EXPECT_NEAR(out_host[itr], out_ref[itr], std::sqrt(box.original_box_.total_size_) * 1e-13) << itr;
    }
}