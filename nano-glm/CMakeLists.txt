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

# Vulkan experts in the backend (PLAN.md step 3, increment 2). OFF by default
# and built in a *separate tree* (build.ps1 -Vk -> build-vk/), for a reason that
# is structural rather than tidiness:
#
# `init_state` aborts when any GPU device is registered (lib/nano_graph.h) —
# the guard that exists because of the AMD-Metal NaN incident. Every trunk
# binary calls ggml_backend_load_all(), so in a Vulkan-enabled build they would
# all see the Vega dies and abort on startup. moe-server does not include
# nano_graph.h and is therefore the only binary that can hold a GPU.
#
# So this tree builds *only* moe-server, and it is paired with the CPU tree's
# client: `gate.py rpc --external-server` against a manually started
# build-vk\bin\moe-server.exe. The client keeps exactly the numerics the golden
# set was made with, which is what makes the comparison mean anything.
option(NANO_VULKAN "build moe-server with the ggml Vulkan backend (separate tree)" OFF)
if(NANO_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

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

# Routing trace (lib/expert_trace.h): per-position, per-layer selected expert
# ids. OFF by default and compiled out entirely — it keeps an intermediate
# tensor alive, which changes graph allocation, and the default build is the
# one the bit-exactness gate runs against. Configure a *separate* build tree
# for it (build.ps1 -Trace) so both binaries exist side by side and can be
# byte-compared.
option(NANO_EXPERT_TRACE "collect per-token expert routing traces" OFF)

# Build fingerprint (lib/build_info.h): the facts that decide whether two runs
# can be expected to agree bit for bit. Printed by --version, recorded in the
# gate's provenance sidecar, and sent over the wire in the protocol handshake.
#
# Captured at *configure* time, so it goes stale if you commit and rebuild
# without reconfiguring — see build_info.h for why that is the deliberate
# choice and what cross-checks it. The dirty flag is scoped to the sources that
# actually go into these binaries, so an edit to some other subproject's README
# does not mark the build dirty and drain the field of meaning.
set(NANO_GIT_REV "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE NANO_GIT_REV
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT NANO_GIT_REV)
        set(NANO_GIT_REV "unknown")
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} status --porcelain --
                ${CMAKE_CURRENT_SOURCE_DIR}/lib
                ${CMAKE_CURRENT_SOURCE_DIR}/apps
                ${CMAKE_CURRENT_SOURCE_DIR}/../logit-kld/src
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE NANO_GIT_DIRTY
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NANO_GIT_DIRTY)
        set(NANO_GIT_REV "${NANO_GIT_REV}-dirty")
    endif()
endif()

add_subdirectory(${LLAMA_CPP_DIR}/ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)

# lib/ is the engine every app shares: model definition and loader, the routed
# expert block, the wire protocol, the build fingerprint, the routing trace.
# An INTERFACE target rather than a compiled one because it is header-only
# today; when an app grows past a single translation unit, the headers with
# mutable state have to become a real static library first (see lib/README.md).
#
# logit-kld/src comes along because logits_file.{h,cpp} is the lkldtopk
# writer/reader — reused so nano-glm output can be compare.py'd straight
# against the llama.cpp collect baselines — as are cpu_topology.h and
# topk_utils.h.
add_library(nano-lib INTERFACE)
target_include_directories(nano-lib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../logit-kld/src
)
target_link_libraries(nano-lib INTERFACE ggml)
target_compile_features(nano-lib INTERFACE cxx_std_17)
if(WIN32)
    target_link_libraries(nano-lib INTERFACE ws2_32)   # moe_proto.h speaks sockets
endif()

# The MoE backend: same model definition and same routed-expert graph as the
# client, reached over TCP instead of a function call. The only target built in
# the Vulkan tree, per the comment on NANO_VULKAN above.
add_executable(moe-server apps/moe-server/main.cpp)
target_link_libraries(moe-server PRIVATE nano-lib)

# Loads a model with the real loader and reports its structure and expert
# footprint. Built in both trees: in the Vulkan one it can also say how much of
# the model the GPUs would hold, which is the question that decides whether
# offload is worth measuring for a given model.
add_executable(nano-probe apps/nano-probe/main.cpp)
target_link_libraries(nano-probe PRIVATE nano-lib)

# The expert kernel with synthetic weights and no model at all. moe-server needs
# ~3 minutes to load 150 GiB and fill four dies before it can be timed, which is
# the wrong edit cycle for tuning a matmul. Built only in the Vulkan tree because
# without a GPU there is nothing here to measure.
if(NANO_VULKAN)
    add_executable(moe-bench apps/moe-bench/main.cpp)
    target_link_libraries(moe-bench PRIVATE nano-lib)
endif()

set(NANO_TARGETS moe-server nano-probe)
if(NANO_VULKAN)
    list(APPEND NANO_TARGETS moe-bench)
endif()

if(NOT NANO_VULKAN)
    # The validation harness: raw token ids in, an lkldtopk file out. Frozen
    # interface on purpose — see PLAN.md step 7 and TESTING.md.
    add_executable(nano-glm
        apps/nano-glm/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../logit-kld/src/logits_file.cpp
    )
    target_link_libraries(nano-glm PRIVATE nano-lib)

    # Throughput in a named residency regime (PLAN.md step 11). Links
    # logits_file.cpp for prompt_source.h, which reads prompts out of lkldtopk
    # files so a bench run and a gate run can be given the identical token ids.
    add_executable(nano-bench
        apps/nano-bench/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../logit-kld/src/logits_file.cpp
    )
    target_link_libraries(nano-bench PRIVATE nano-lib)

    # Single-turn chat: tokenizer, chat template, streaming text. Everything
    # that can change a token sequence lives here rather than in nano-glm, so
    # that adding a sampler cannot invalidate the golden set.
    add_executable(nano-chat apps/nano-chat/main.cpp)
    target_link_libraries(nano-chat PRIVATE nano-lib)

    # Temporary: the deepseek4 porting harness. Runs the trunk as far as it is
    # ported and dumps named tensors for comparison against llama.cpp
    # (logit-kld's `dump`). Goes away when the graph is complete.
    add_executable(ds4-port apps/ds4-port/main.cpp)
    target_link_libraries(ds4-port PRIVATE nano-lib)

    list(APPEND NANO_TARGETS nano-glm nano-bench nano-chat ds4-port)
endif()

# moe_block.h is shared, so both binaries must agree on whether it traces.
if(NANO_EXPERT_TRACE)
    foreach(tgt ${NANO_TARGETS})
        target_compile_definitions(${tgt} PRIVATE NANO_EXPERT_TRACE)
    endforeach()
    message(STATUS "nano-glm: expert routing trace ENABLED (--expert-log)")
endif()

# Compiler id and version go in as two defines rather than one joined string:
# neither contains a space, which sidesteps quoting a -D value through cmake,
# ninja and cl. build_info.h joins them.
foreach(tgt ${NANO_TARGETS})
    target_compile_definitions(${tgt} PRIVATE
        NANO_GIT_REV="${NANO_GIT_REV}"
        NANO_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
        NANO_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
        NANO_BUILD_BLAS=$<IF:$<BOOL:${GGML_BLAS}>,1,0>
        NANO_BUILD_LLAMAFILE=$<IF:$<BOOL:${GGML_LLAMAFILE}>,1,0>
        NANO_BUILD_VULKAN=$<IF:$<BOOL:${NANO_VULKAN}>,1,0>
    )
endforeach()
message(STATUS "nano-glm: build ${NANO_GIT_REV} / ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(NANO_VULKAN)
    message(STATUS "nano-glm: VULKAN backend ENABLED — moe-server only")
endif()

if(WIN32)
    # ggml builds DLLs into ${CMAKE_BINARY_DIR}/bin; Windows resolves them from
    # the exe's own directory, so put the exes there too rather than making every
    # invocation prepend build/bin to PATH.
    set_target_properties(${NANO_TARGETS} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
