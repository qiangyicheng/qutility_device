set(CURRENT_TARGET_NAME math)


target_link_libraries(qutility_device_${CURRENT_TARGET_NAME} INTERFACE
    qutility_device_sync_grid
)

target_sources(qutility_device_${CURRENT_TARGET_NAME} INTERFACE
    elementwise.cuh
    eigenvalue.cuh
)