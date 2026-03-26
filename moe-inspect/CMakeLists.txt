cmake_minimum_required(VERSION 3.14)
project(moe-inspect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to llama.cpp source and build directories.
# Override with: cmake -DLLAMA_CPP_DIR=... -DLLAMA_CPP_BUILD_DIR=...
# Falls back to LLAMA_CPP_DIR environment variable.
set(LLAMA_CPP_DIR "$ENV{LLAMA_CPP_DIR}" CACHE PATH "Path to llama.cpp source")
if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "LLAMA_CPP_DIR is not set. Pass -DLLAMA_CPP_DIR=... or set the LLAMA_CPP_DIR environment variable.")
endif()
set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_DIR}/build" CACHE PATH "Path to llama.cpp build directory")

# ggml headers
include_directories(${LLAMA_CPP_DIR}/ggml/include)

# Link against libggml-base (contains gguf + ggml core)
find_library(GGML_BASE_LIB
    NAMES ggml-base
    PATHS ${LLAMA_CPP_BUILD_DIR}/bin
    NO_DEFAULT_PATH
    REQUIRED
)

message(STATUS "Using llama.cpp source: ${LLAMA_CPP_DIR}")
message(STATUS "Using ggml-base lib:    ${GGML_BASE_LIB}")

# --- moe-eval-size ---
add_executable(moe-eval-size moe-eval-size.cpp)
target_link_libraries(moe-eval-size PRIVATE ${GGML_BASE_LIB})

# Set rpath so the binary finds the dylib at runtime
set_target_properties(moe-eval-size PROPERTIES
    BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    INSTALL_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
)
