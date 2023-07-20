set(CURRENT_TARGET_NAME workspace)

target_link_libraries(qutility_device_${CURRENT_TARGET_NAME} PRIVATE qutility)

if(QUTILITY_DEVICE_USE_CUDA)
    target_link_libraries(qutility_device_${CURRENT_TARGET_NAME}
        PRIVATE
        CUDA::cudart_static
        CUDA::cudart
        CUDA::culibos
    )

# propogate the property to TARGET_DIRECTORY of qutility_device_${CURRENT_TARGET_NAME}
set_property(SOURCE ${CURRENT_TARGET_NAME}.cpp TARGET_DIRECTORY qutility_device_${CURRENT_TARGET_NAME} PROPERTY LANGUAGE CUDA)
else()
    find_package(hip)
    target_link_libraries(qutility_device_${CURRENT_TARGET_NAME}
        PRIVATE
        hip::host
        hip::device
    )
endif()

target_sources(qutility_device_${CURRENT_TARGET_NAME} PRIVATE
    ${CURRENT_TARGET_NAME}.cpp
)
