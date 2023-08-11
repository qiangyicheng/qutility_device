#include "qutility_device/fft_cb/cb.cuh"

namespace qutility
{
    namespace device
    {
        namespace fft_cb
        {
            namespace kernel
            {
                
                /// @brief Load an array from another address
                __device__ dapi_cufftCallbackLoadD D2ZLoadSideLoadDevicePtr = D2ZLoadSideLoad;

                /// @brief Multiple an array while loading of D2Z transform
                __device__ dapi_cufftCallbackLoadD D2ZLoadMulDevicePtr = D2ZLoadMul;

                /// @brief Shift the reading address of the input data and multiple an array while loading of D2Z transform
                __device__ dapi_cufftCallbackLoadD D2ZLoadShiftLoadMulDevicePtr = D2ZLoadShiftLoadMul;

                /// @brief Multiple an array and add it to the dst, while multiple another array while loading of D2Z transform
                __device__ dapi_cufftCallbackLoadD D2ZLoadHalfMulIncreMulDevicePtr = D2ZLoadHalfMulIncreMul;

                /// @brief Multiple an array while storing of D2Z transform
                ///        !***CAUTION***!: the callerInfo is a ****REAL**** matrix that has the same data layout, thus the number of elements, with dataOut, which is a complex matrix with Hermit symmetry
                __device__ dapi_cufftCallbackStoreZ D2ZStoreMulRealDevicePtr = D2ZStoreMulReal;

                /// @brief Multiple a complex array while storing of D2Z transform, and increment another array by the imaginary part of the result
                __device__ dapi_cufftCallbackStoreZ D2ZStoreMulComplexDropRealIncreDevicePtr = D2ZStoreMulComplexDropRealIncre;

                /// @brief Multiple a complex array while storing of D2Z transform, and increment another array by the imaginary part of the result
                __device__ dapi_cufftCallbackStoreZ D2ZStoreHalfMulComplexDropRealIncreDevicePtr = D2ZStoreHalfMulComplexDropRealIncre;

                /// @brief Multiple an complex array while loading of Z2D transform
                __device__ dapi_cufftCallbackLoadZ Z2DLoadMulComplexDevicePtr = Z2DLoadMulComplex;

                /// @brief Multiple an array while storing of Z2D transform
                __device__ dapi_cufftCallbackStoreD Z2DStoreMulDevicePtr = Z2DStoreMul;

                /// @brief Multiple an array while storing of Z2D transform, while multiple another array and increment the third array by the result
                __device__ dapi_cufftCallbackStoreD Z2DStoreMulMulIncreDevicePtr = Z2DStoreMulMulIncre;

                /// @brief Multiple an array while storing of Z2D transform, while multiple another array and increment the third array by half of the result
                __device__ dapi_cufftCallbackStoreD Z2DStoreMulHalfMulIncreDevicePtr = Z2DStoreMulHalfMulIncre;

            }
        }
    }
}
