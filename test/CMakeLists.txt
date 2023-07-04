# Include Google Test

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/2f80c2ba71c0e8922a03b9b855e5b019ad1f7064.zip
)

# Include Qutility

FetchContent_Declare(qutility
    GIT_REPOSITORY    https://github.com/qiangyicheng/qutility_cmake.git
    GIT_TAG           main
)

if(NOT qutility_POPULATED)
    FetchContent_Populate(qutility)
    add_subdirectory(${qutility_SOURCE_DIR} ${qutility_BINARY_DIR})
endif()


# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
set(TEST_NAME info)

if(QUTILITY_DEVICE_USE_CUDA)
    set_property(SOURCE ${TEST_NAME}_test.cpp PROPERTY LANGUAGE CUDA)
endif()
add_executable(
    ${TEST_NAME}_test
    ${TEST_NAME}_test.cpp
)

target_link_libraries(${TEST_NAME}_test
    PRIVATE
    ${PROJECT_NAME}_${TEST_NAME}
    qutility
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(${TEST_NAME}_test)
