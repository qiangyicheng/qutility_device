add_subdirectory(info)
add_subdirectory(event)


##dummy
target_sources(${PROJECT_NAME}
    PRIVATE
    dummy.cpp
)