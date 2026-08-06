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
set(GGML_METAL       OFF     CACHE BOOL   "" FORCE)
set(GGML_BLAS        ON      CACHE BOOL   "" FORCE)
set(GGML_BLAS_VENDOR "Apple" CACHE STRING "" FORCE)
set(GGML_ACCELERATE  ON      CACHE BOOL   "" FORCE)
set(GGML_LLAMAFILE   ON      CACHE BOOL   "" FORCE)

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
