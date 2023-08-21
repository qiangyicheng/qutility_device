# Include Google Test

include(FetchContent)
FetchContent_Declare(
    googletest
  GIT_REPOSITORY git@github.com:google/googletest.git
  GIT_TAG main
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

# test sync_grid_atomic
set(TEST_NAME sync_grid_atomic)

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
    ${PROJECT_NAME}_sync_grid
    ${PROJECT_NAME}_event
    ${PROJECT_NAME}_workspace
    device_api
    qutility
    gtest_main
)
target_compile_definitions(${TEST_NAME}_test
    PRIVATE
    QUTILITY_DEVICE_USE_ATOMIC_GRID_SYNC
)
gtest_discover_tests(${TEST_NAME}_test)

# test field
set(TEST_NAME field)

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
    device_api
    qutility
    gtest_main
)
gtest_discover_tests(${TEST_NAME}_test)

# test fft_cb
set(TEST_NAME fft_cb)

if(QUTILITY_DEVICE_USE_CUDA)
    set_property(SOURCE ${TEST_NAME}_test.cpp PROPERTY LANGUAGE CUDA)
else()
endif()

add_executable(
    ${TEST_NAME}_test
    ${TEST_NAME}_test.cpp
)

if(QUTILITY_DEVICE_USE_CUDA)
    set_property(TARGET ${TEST_NAME}_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
else()
endif()

find_package(FFTW3 CONFIG REQUIRED)

target_link_libraries(${TEST_NAME}_test
    PRIVATE
    ${PROJECT_NAME}_${TEST_NAME}
    device_api
    qutility
    gtest_main
    FFTW3::fftw3
)
gtest_discover_tests(${TEST_NAME}_test)

# test math
set(TEST_NAME math)

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
    ${PROJECT_NAME}_field
    device_api
    qutility
    gtest_main
)
gtest_discover_tests(${TEST_NAME}_test)
