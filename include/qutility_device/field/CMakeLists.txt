set(CURRENT_TARGET_NAME field)

target_sources(qutility_device_${CURRENT_TARGET_NAME} INTERFACE
    ${CURRENT_TARGET_NAME}.h
)

target_link_libraries(qutility_device_${CURRENT_TARGET_NAME} INTERFACE
    qutility
    qutility_device_event
)