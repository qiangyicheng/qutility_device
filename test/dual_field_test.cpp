// c++ headers
#include <iostream>

// gtest headers
#include <gtest/gtest.h>

// project headers
#include <qutility_device/dual_field.h>

TEST(QutilityDeviceDualHistory, SetValue)
{
    using ValT = double;
    constexpr int device = 0;
    constexpr std::size_t size = 1024 * 8;
    qutility::device::dual_field::DualFieldEx<ValT> df(size, device);

    for (std::size_t itr = 0; itr < size; ++itr)
    {
        df.field_host_[itr] = (double)rand() / RAND_MAX;
        df.field_diff_host_[itr] = (double)rand() / RAND_MAX;
    }
    df.host_to_device();
    df.sync_device();

    EXPECT_EQ(df.field_[size / 2], df.field_host_[size / 2]);
    EXPECT_EQ(df.field_diff_[size / 2], df.field_diff_host_[size / 2]);
}