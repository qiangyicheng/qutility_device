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

                /// Multiple an array while storing of D2Z transform
                /// !***CAUTION***!: the callerInfo is a ****REAL**** matrix that has the same data layout, thus the number of elements, with dataOut, which is a complex matrix with Hermit symmetry
                extern __device__ cufftCallbackStoreZ D2ZStoreMulRealDevicePtr = D2ZStoreMulReal;

            }
        }
    }
}
