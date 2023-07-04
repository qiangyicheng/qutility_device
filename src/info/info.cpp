#include "qutility_device/info/info.h"

#include "device_api/device_api_cuda_runtime.h"
#include "device_api/device_api_helper.h"

namespace qutility
{
    namespace device
    {
        namespace info
        {
            auto ptr_device_id(const void *ptr) -> int
            {
                dapi_cudaPointerAttributes attr;
                dapi_checkCudaErrors(dapi_cudaPointerGetAttributes(&attr, ptr));
                if (dapi_cudaMemoryTypeDevice != attr.type)
                    return -1;
                return attr.device;
            }

            auto device_max_blocks_and_threads(int device) -> std::pair<std::size_t, std::size_t>
            {
                std::size_t max_blocks, max_threads;
                dapi_cudaDeviceProp prop = {0};
                dapi_checkCudaErrors(dapi_cudaSetDevice(device));
                dapi_checkCudaErrors(dapi_cudaGetDeviceProperties(&prop, device));
                max_threads = prop.maxThreadsPerBlock;
#ifndef QUTILITY_DEVICE_USE_HIP
                max_blocks = (size_t)(prop.multiProcessorCount) * (prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);
#else
                max_blocks = (size_t)(prop.multiProcessorCount); // prop.maxThreadsPerMultiProcessor always gives 0 in HIP for AMD platform
#endif // QUTILITY_DEVICE_USE_HIP
                return {max_blocks, max_threads};
            }

            auto device_min_nice_threads(int device) -> std::size_t
            {
#ifndef QUTILITY_DEVICE_USE_HIP
                dapi_cudaDeviceProp prop = {0};
                dapi_checkCudaErrors(dapi_cudaSetDevice(device));
                dapi_checkCudaErrors(dapi_cudaGetDeviceProperties(&prop, device));
                return _ConvertSMVer2Cores(prop.major, prop.minor); //this function is dedicated to cuda_helper.h
#else
                return 128; //TODO: obtain number of cores for AMD platform
#endif // QUTILITY_DEVICE_USE_HIP
            }
        }
    }
}
