set(CURRENT_TARGET_NAME def)

target_sources(qutility_device_${CURRENT_TARGET_NAME} INTERFACE
    ${CURRENT_TARGET_NAME}.h
    config.h
)