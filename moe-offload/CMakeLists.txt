cmake_minimum_required(VERSION 3.16)
project(moe-offload LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vulkan only — no ggml, no llama.cpp. The point is to measure the driver, so
# the fewer layers between this code and it, the better.
find_package(Vulkan REQUIRED)

# glslc ships with the Vulkan SDK on both platforms; on macOS it may also come
# from `brew install shaderc`.
find_program(GLSLC NAMES glslc
             HINTS ENV VULKAN_SDK
             PATH_SUFFIXES bin Bin)
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK (or `brew install shaderc`) "
                        "and ensure VULKAN_SDK is set.")
endif()

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SPV_DIR    ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPV_DIR})

set(SHADERS matmul_i8 silu_mul nullk)
set(SPV_FILES "")
foreach(s ${SHADERS})
    add_custom_command(
        OUTPUT  ${SPV_DIR}/${s}.spv
        COMMAND ${GLSLC} -O ${SHADER_DIR}/${s}.comp -o ${SPV_DIR}/${s}.spv
        DEPENDS ${SHADER_DIR}/${s}.comp
        COMMENT "glslc ${s}.comp")
    list(APPEND SPV_FILES ${SPV_DIR}/${s}.spv)
endforeach()
add_custom_target(moe-offload-shaders DEPENDS ${SPV_FILES})

add_executable(moe-offload src/moe_offload.cpp)
add_dependencies(moe-offload moe-offload-shaders)
target_link_libraries(moe-offload PRIVATE Vulkan::Vulkan)
target_include_directories(moe-offload PRIVATE src)

# The binary loads .spv at runtime; tell it where they landed.
target_compile_definitions(moe-offload PRIVATE MOE_SPV_DIR="${SPV_DIR}")
