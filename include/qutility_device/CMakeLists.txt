
## qutility_device::info
set(CURRENT_TARGET_NAME qutility_device_info)
add_library(${CURRENT_TARGET_NAME} STATIC)
# set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(${CURRENT_TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include/)
target_sources(${CURRENT_TARGET_NAME} PUBLIC
    info.h
)


add_subdirectory(info)