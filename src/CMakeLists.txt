add_subdirectory(info)
add_subdirectory(event)
add_subdirectory(workspace)


##dummy
target_sources(${PROJECT_NAME}
    PRIVATE
    ${PROJECT_NAME}.cpp
)