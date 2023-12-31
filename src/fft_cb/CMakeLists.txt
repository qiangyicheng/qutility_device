set(CURRENT_TARGET_NAME fft_cb)

target_link_libraries(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PRIVATE device_api)

if(QUTILITY_DEVICE_USE_CUDA)
    target_link_libraries(${PROJECT_NAME}_${CURRENT_TARGET_NAME}
        PUBLIC
        CUDA::cudart_static
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::culibos
        CUDA::cufft_static
    )

    if(EXISTS "${CUDAToolkit_BIN_DIR}/../samples/common/inc/helper_cuda.h")
        message(STATUS "Header helper_cuda.h is found in ${CUDAToolkit_BIN_DIR}/../samples/common/inc/")
        target_include_directories(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PRIVATE ${CUDAToolkit_BIN_DIR}/../samples/common/inc/)
    else()
        message(STATUS "Header helper_cuda.h is not found in ${CUDAToolkit_BIN_DIR}/../samples/common/inc/. Fetch cuda-samples")

        FetchContent_Declare(cuda-samples
            GIT_REPOSITORY git@github.com:NVIDIA/cuda-samples.git
            GIT_TAG master
        )

        FetchContent_GetProperties(cuda-samples)

        if(NOT cuda-samples_POPULATED)
            FetchContent_Populate(cuda-samples)
            message(STATUS "use cuda-samples headers in ${cuda-samples_SOURCE_DIR}/Common")
        endif()

        target_include_directories(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PRIVATE ${cuda-samples_SOURCE_DIR}/Common)
    endif()

else()
    target_compile_definitions(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PRIVATE QUTILITY_DEVICE_USE_HIP)
    find_package(hip)
    target_link_libraries(${PROJECT_NAME}_${CURRENT_TARGET_NAME}
        PUBLIC
        hip::host
        hip::device
        hip::hipfft
    )
endif()

target_sources(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PRIVATE
    fft.cpp
    cb.cu
)
