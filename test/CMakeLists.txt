# Include Google Test

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/2f80c2ba71c0e8922a03b9b855e5b019ad1f7064.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# test info
set(TEST_NAME info)

if(QUTILITY_DEVICE_USE_CUDA)
    set_property(SOURCE ${TEST_NAME}_test.cpp PROPERTY LANGUAGE CUDA)
else()
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
gtest_discover_tests(${TEST_NAME}_test)

# test event
set(TEST_NAME event)

if(QUTILITY_DEVICE_USE_CUDA)
    set_property(SOURCE ${TEST_NAME}_test.cpp PROPERTY LANGUAGE CUDA)
else()
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
gtest_discover_tests(${TEST_NAME}_test)

# test sync_grid
set(TEST_NAME sync_grid)

if(QUTILITY_DEVICE_USE_CUDA)
    set_property(SOURCE ${TEST_NAME}_test.cpp PROPERTY LANGUAGE CUDA)
else()
endif()

add_executable(
    ${TEST_NAME}_test
    ${TEST_NAME}_test.cpp
)
target_link_libraries(${TEST_NAME}_test
    PRIVATE
    ${PROJECT_NAME}_${TEST_NAME}
    ${PROJECT_NAME}_event
    device_api
    qutility
    gtest_main
)
gtest_discover_tests(${TEST_NAME}_test)
