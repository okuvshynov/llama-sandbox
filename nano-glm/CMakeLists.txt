cmake_minimum_required(VERSION 3.14)
project(nano-glm LANGUAGES C CXX)

set(LLAMA_CPP_DIR "$ENV{LLAMA_CPP_DIR}" CACHE PATH "Path to llama.cpp source tree (only ggml/ is used)")
if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "LLAMA_CPP_DIR is not set. Pass -DLLAMA_CPP_DIR=... or set the LLAMA_CPP_DIR environment variable.")
endif()

if(NOT EXISTS "${LLAMA_CPP_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR "ggml not found at ${LLAMA_CPP_DIR}/ggml")
endif()

# CPU-only on purpose (see repo CLAUDE.md: AMD-Metal op offload silently NaNs
# logits at decode batches >= 32 on this machine). The remaining options mirror
# the logit-kld / llama.cpp build exactly — BLAS (Accelerate) and llamafile
# sgemm are numerics-relevant, and the whole point of nano-glm is bit-identical
# logits vs the llama.cpp baseline.
set(GGML_METAL     OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON  CACHE BOOL "" FORCE)

# BLAS is what the macOS baselines ran with (Accelerate, engaged at batch >= 32
# via ggml_backend_sched op offload), so it stays ON there. Off-Apple there is
# no equivalent to match bit-for-bit: a different BLAS is a different set of
# numerics, so the honest default is no BLAS at all — every op runs in ggml's
# own CPU kernels. Consequence: batches >= 32 (i.e. prompt prefill) are NOT
# expected to reproduce the macOS baseline bit-for-bit; the decode path, which
# never crosses the threshold, still is. Override with -DGGML_BLAS=ON to
# experiment.
if(APPLE)
    set(GGML_BLAS        ON      CACHE BOOL   "" FORCE)
    set(GGML_BLAS_VENDOR "Apple" CACHE STRING "" FORCE)
    set(GGML_ACCELERATE  ON      CACHE BOOL   "" FORCE)
else()
    set(GGML_BLAS        OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(${LLAMA_CPP_DIR}/ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)

# logits_file.{h,cpp} is logit-kld's lkldtopk writer/reader, reused so nano-glm
# output can be compare.py'd directly against the llama.cpp collect baselines.
add_executable(nano-glm
    src/nano_glm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../logit-kld/src/logits_file.cpp
)

target_include_directories(nano-glm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../logit-kld/src
)

target_link_libraries(nano-glm PRIVATE ggml)
target_compile_features(nano-glm PRIVATE cxx_std_17)

# The MoE backend: same model definition and same routed-expert graph
# (nano_model.h / moe_block.h), reached over TCP instead of a function call.
add_executable(moe-server src/moe_server.cpp)
target_include_directories(moe-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../logit-kld/src
)
target_link_libraries(moe-server PRIVATE ggml)
target_compile_features(moe-server PRIVATE cxx_std_17)
if(WIN32)
    target_link_libraries(moe-server PRIVATE ws2_32)
endif()

if(WIN32)
    # ggml builds DLLs into ${CMAKE_BINARY_DIR}/bin; Windows resolves them from
    # the exe's own directory, so put the exes there too rather than making every
    # invocation prepend build/bin to PATH.
    set_target_properties(nano-glm moe-server PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
