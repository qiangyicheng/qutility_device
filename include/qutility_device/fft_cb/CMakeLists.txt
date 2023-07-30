set(CURRENT_TARGET_NAME fft_cb)

find_package(Boost 1.61 REQUIRED)

target_link_libraries(qutility_device_${CURRENT_TARGET_NAME} PUBLIC
    qutility
    qutility_device_event
    qutility_device_workspace
    device_api
    Boost::headers
)

target_sources(qutility_device_${CURRENT_TARGET_NAME} PUBLIC
    ${CURRENT_TARGET_NAME}.cuh
    fft.h
    cb.cuh
)