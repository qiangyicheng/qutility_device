#include "qutility_device/fft_cb/cb.cuh"

namespace qutility
{
    namespace device
    {
        namespace fft_cb
        {
            namespace kernel
            {
                /// Multiple an array while loading of D2Z transform
                __device__ dapi_cufftCallbackLoadD D2ZLoadMulDevicePtr = D2ZLoadMul;

                /// Shift the reading address of the input data and multiple an array while loading of D2Z transform
                __device__ dapi_cufftCallbackLoadD D2ZLoadShiftLoadMulDevicePtr = D2ZLoadShiftLoadMul;

                /// Multiple an array and add it to the dst, while multiple another array while loading of D2Z transform
                __device__ dapi_cufftCallbackLoadD D2ZLoadHalfMulIncreMulDevicePtr = D2ZLoadHalfMulIncreMul;

                /// Multiple an array while storing of D2Z transform
                /// !***CAUTION***!: the callerInfo is a ****REAL**** matrix that has the same data layout, thus the number of elements, with dataOut, which is a complex matrix with Hermit symmetry
                __device__ dapi_cufftCallbackStoreZ D2ZStoreMulRealDevicePtr = D2ZStoreMulReal;

                /// Multiple an complex array while loading of Z2D transform
                __device__ dapi_cufftCallbackLoadZ Z2DLoadMulComplexDevicePtr = Z2DLoadMulComplex;

                /// Multiple an array while storing of Z2D transform
                __device__ dapi_cufftCallbackStoreD Z2DStoreMulDevicePtr = Z2DStoreMul;

                __device__ dapi_cufftCallbackStoreD Z2DStoreMulMulIncreDevicePtr = Z2DStoreMulMulIncre;

                __device__ dapi_cufftCallbackStoreD Z2DStoreMulHalfMulIncreDevicePtr = Z2DStoreMulHalfMulIncre;

            }
        }
    }
}
