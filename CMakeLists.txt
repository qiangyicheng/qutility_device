# CMakeLists files in this project can
# refer to the root source directory of the project as ${PROJECT_SOURCE_DIR} and
# to the root binary directory of the project as ${PROJECT_BINARY_DIR}.

cmake_minimum_required(VERSION 3.14)

include(CheckLanguage)
check_language(CUDA)
set(QUTILITY_DEVICE_USE_CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA is found. This project will target CUDA platform directly")
    set(QUTILITY_DEVICE_USE_CUDA True)
    project(qutility_device LANGUAGES CUDA CXX)
    set(CMAKE_CUDA_ARCHITECTURES "86")
    find_package(CUDAToolkit REQUIRED)
else()
    message(STATUS "CUDA is not found. This prpject will try HIP platform")
    set(QUTILITY_DEVICE_USE_CUDA False)
    project(qutility_device LANGUAGES CXX)
    find_package(hip REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(device_api
    GIT_REPOSITORY    https://github.com/qiangyicheng/device_api.git
)
FetchContent_GetProperties(device_api)
if(NOT device_api_POPULATED)
    FetchContent_Populate(device_api)
    add_subdirectory(${device_api_SOURCE_DIR} ${device_api_BINARY_DIR})
endif()

add_library(${PROJECT_NAME} STATIC)
target_include_directories(${PROJECT_NAME} PRIVATE include/)

add_subdirectory(include)
add_subdirectory(src)

enable_testing()
add_subdirectory(test)