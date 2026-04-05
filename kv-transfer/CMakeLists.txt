cmake_minimum_required(VERSION 3.14)
project(kv-transfer LANGUAGES C CXX)

set(LLAMA_CPP_DIR "$ENV{LLAMA_CPP_DIR}" CACHE PATH "Path to llama.cpp source tree")
if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "LLAMA_CPP_DIR is not set. Pass -DLLAMA_CPP_DIR=... or set the LLAMA_CPP_DIR environment variable.")
endif()

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp not found at ${LLAMA_CPP_DIR}")
endif()

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)

find_package(Threads REQUIRED)

add_executable(kv-transfer
    src/main.cpp
    src/trace_io.cpp
    src/stats_io.cpp
    src/ref.cpp
    src/target.cpp
    src/handoff.cpp
    src/compare.cpp
    src/decay.cpp
)

target_include_directories(kv-transfer PRIVATE
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
)

target_link_libraries(kv-transfer PRIVATE
    common
    llama
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_features(kv-transfer PRIVATE cxx_std_17)
