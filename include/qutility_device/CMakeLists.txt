# # qutility_device_info
set(CURRENT_TARGET_NAME info)
add_library(qutility_device_${CURRENT_TARGET_NAME} STATIC)
target_include_directories(qutility_device_${CURRENT_TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include/)
target_sources(qutility_device_${CURRENT_TARGET_NAME} PUBLIC
    ${CURRENT_TARGET_NAME}.h
)

add_subdirectory(${CURRENT_TARGET_NAME})

# # qutility_device_event
set(CURRENT_TARGET_NAME event)
add_library(qutility_device_${CURRENT_TARGET_NAME} STATIC)
target_include_directories(qutility_device_${CURRENT_TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include/)
target_sources(qutility_device_${CURRENT_TARGET_NAME} PUBLIC
    ${CURRENT_TARGET_NAME}.h
)

add_subdirectory(${CURRENT_TARGET_NAME})

# # qutility_device_def
set(CURRENT_TARGET_NAME def)
add_library(qutility_device_${CURRENT_TARGET_NAME} INTERFACE)
target_include_directories(qutility_device_${CURRENT_TARGET_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include/)
target_sources(qutility_device_${CURRENT_TARGET_NAME} INTERFACE
    ${CURRENT_TARGET_NAME}.h
)

add_subdirectory(${CURRENT_TARGET_NAME})

# # qutility_device_sync_grid
set(CURRENT_TARGET_NAME sync_grid)
add_library(qutility_device_${CURRENT_TARGET_NAME} INTERFACE)
target_include_directories(qutility_device_${CURRENT_TARGET_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include/)
target_sources(qutility_device_${CURRENT_TARGET_NAME} INTERFACE
    ${CURRENT_TARGET_NAME}.cuh
)

add_subdirectory(${CURRENT_TARGET_NAME})

# # qutility_device_workspace
set(CURRENT_TARGET_NAME workspace)
add_library(qutility_device_${CURRENT_TARGET_NAME} STATIC)
target_include_directories(qutility_device_${CURRENT_TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include/)
target_sources(qutility_device_${CURRENT_TARGET_NAME} PUBLIC
    ${CURRENT_TARGET_NAME}.h
)

add_subdirectory(${CURRENT_TARGET_NAME})
