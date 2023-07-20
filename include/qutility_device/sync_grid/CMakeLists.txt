set(CURRENT_TARGET_NAME sync_grid)

target_link_libraries(qutility_device_${CURRENT_TARGET_NAME} INTERFACE
    qutility_device_def
    device_api
)

if(QUTILITY_DEVICE_USE_CUDA)
    target_link_libraries(qutility_device_${CURRENT_TARGET_NAME}
        INTERFACE
        CUDA::cuda_driver
    )
else()
    find_package(hip)
    target_link_libraries(qutility_device_${CURRENT_TARGET_NAME}
        INTERFACE
        hip::device
    )
endif()

target_sources(qutility_device_${CURRENT_TARGET_NAME} INTERFACE
    ${CURRENT_TARGET_NAME}.cuh
)