set(CURRENT_TARGET_NAME workspace)

target_sources(qutility_device_${CURRENT_TARGET_NAME} PUBLIC
    ${CURRENT_TARGET_NAME}.h
)

target_link_libraries(qutility_device_${CURRENT_TARGET_NAME} PUBLIC
    qutility_device_def
    qutility_device_event
)