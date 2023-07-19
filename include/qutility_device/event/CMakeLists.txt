set(CURRENT_TARGET_NAME event)

target_link_libraries(qutility_device_${CURRENT_TARGET_NAME} PUBLIC device_api)
target_link_libraries(qutility_device_${CURRENT_TARGET_NAME} PUBLIC qutility)

target_sources(qutility_device_${CURRENT_TARGET_NAME} PUBLIC
    ${CURRENT_TARGET_NAME}.h
)