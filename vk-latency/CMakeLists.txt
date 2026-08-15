cmake_minimum_required(VERSION 3.14)
project(vk-latency LANGUAGES CXX)

# Deliberately no ggml, no llama.cpp: this measures the Vulkan border with
# nothing of ours around it, so moe-serv's numbers have a machine baseline.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Vulkan REQUIRED)

add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/bin/shaders/null.spv"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin/shaders"
    COMMAND Vulkan::glslc -O -o "${CMAKE_BINARY_DIR}/bin/shaders/null.spv"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/null.comp"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/null.comp"
    COMMENT "glslc null.comp")
add_custom_target(vk-latency-shaders DEPENDS "${CMAKE_BINARY_DIR}/bin/shaders/null.spv")

add_executable(vk-latency src/main.cpp)
add_dependencies(vk-latency vk-latency-shaders)
target_link_libraries(vk-latency PRIVATE Vulkan::Vulkan)
set_target_properties(vk-latency PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
