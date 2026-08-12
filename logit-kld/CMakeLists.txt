cmake_minimum_required(VERSION 3.14)
project(logit-kld LANGUAGES C CXX)

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

# CPU-only on purpose: this is measurement tooling, and on machines with a
# barely-supported GPU (e.g. AMD via Metal on Intel Macs) the default op
# offload at batch sizes >= 32 silently produces NaN logits with ngl=0.
set(GGML_METAL OFF CACHE BOOL "" FORCE)

# Reference *kernels*, not the fastest ones. GGML_CPU_REPACK (ON by default)
# rewrites quantized weights into a blocked layout at load and runs a different
# GEMM against them, which is a different summation order and therefore
# different numbers. nano-glm mmaps its weights as they sit in the file — a
# 583 GiB model cannot afford a converted second copy, and neither can the
# remote-expert design — so it can never match a repacked reference.
#
# This did not surface until DeepSeek-V4-Flash because the repack table is
# per-type *and* per-ISA: GLM-5.2's Q6_K only repacks under NEON, so on this
# x86 machine nothing was ever converted and the golden set was unaffected.
# Deepseek4's MXFP4 experts repack under AVX2, which is present here — 140 GiB
# of `mxfp4_8x8`, and a ~1e-6 disagreement in every routed-expert matmul.
#
# Consequence worth stating: this makes the reference slower, so a *performance*
# comparison against llama.cpp must be built without this line. Only the
# correctness comparison wants it (nano-glm/PLAN.md step 14).
#
# ...and ON is what a *performance* comparison needs, because ON is what
# llama.cpp ships. An option rather than an edit: the two builds have to be
# identical in everything else for either number to mean anything, and an
# edited CMakeLists is not a configuration anyone can point at afterwards.
# Measured on DeepSeek-V4-Flash: repacking is worth +23% prefill and +11%
# decode, which is the handicap nano-glm accepts by mmapping weights as they
# sit in the file (nano-glm/OPTIMIZATION.md).
option(LOGIT_KLD_CPU_REPACK "build the reference tools with llama.cpp's weight repacking" OFF)
set(GGML_CPU_REPACK ${LOGIT_KLD_CPU_REPACK} CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)

find_package(Threads REQUIRED)

# `dump` captures llama.cpp's named intermediates so a port can be checked
# tensor by tensor instead of only at the logits — see src/dump.cpp.
foreach(tool collect rescore dump)
    add_executable(${tool}
        src/${tool}.cpp
        src/logits_file.cpp
    )

    target_include_directories(${tool} PRIVATE
        ${LLAMA_CPP_DIR}/include
        ${LLAMA_CPP_DIR}/common
    )

    target_link_libraries(${tool} PRIVATE
        llama-common
        llama
        ${CMAKE_THREAD_LIBS_INIT}
    )

    target_compile_features(${tool} PRIVATE cxx_std_17)

    if(WIN32)
        # llama/ggml build DLLs into ${CMAKE_BINARY_DIR}/bin; Windows resolves
        # them from the exe's own directory, so put the exes there too.
        set_target_properties(${tool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    endif()
endforeach()
