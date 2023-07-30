add_subdirectory(info)
add_subdirectory(event)
add_subdirectory(workspace)
add_subdirectory(fft_cb)


##dummy
target_sources(${PROJECT_NAME}
    PRIVATE
    ${PROJECT_NAME}.cpp
)