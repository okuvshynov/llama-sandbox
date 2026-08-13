cmake_minimum_required(VERSION 3.14)
project(moe-serv LANGUAGES C CXX)

# Only ggml is used, and only its base library. This project does not link
# llama, does not include llama headers, and never patches llama.cpp — it is
# loaded into whatever llama.cpp binary the user already has, through
# GGML_BACKEND_PATH. If something here ever needs an upstream change, that is a
# finding for PLAN.md, not a diff to carry.
set(LLAMA_CPP_DIR "$ENV{LLAMA_CPP_DIR}" CACHE PATH "Path to llama.cpp source tree (only ggml/ is used)")
if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "LLAMA_CPP_DIR is not set. Pass -DLLAMA_CPP_DIR=... or set the LLAMA_CPP_DIR environment variable.")
endif()
if(NOT EXISTS "${LLAMA_CPP_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR "ggml not found at ${LLAMA_CPP_DIR}/ggml")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# GGML_BACKEND_DL makes ggml's own backends dynamic too. We do not need that —
# we are the only dynamic backend here and we load into a *prebuilt* llama.cpp —
# but the flag also gates GGML_BACKEND_DL_IMPL, which is what generates our
# `ggml_backend_init` / `ggml_backend_score` exports. So it is set as a compile
# definition on our target only, below, rather than as a ggml-wide option.
# Vulkan is built here only for `moe-probe`, which needs a device in its *own*
# process. It changes nothing about moeserv.dll: that links ggml as an import
# library and resolves to the host llama.cpp's copy at load time, so the host's
# build decides what backends the library sees. The cost is shader compilation
# on a clean build.
set(GGML_VULKAN ON CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_DIR}/ggml" ggml EXCLUDE_FROM_ALL)

add_library(moeserv SHARED src/moe_backend.cpp)

# All three are needed and each does a different job:
#   GGML_BACKEND_DL     gates GGML_BACKEND_DL_IMPL, which generates the
#                       ggml_backend_init / ggml_backend_score entry points
#   GGML_BACKEND_SHARED gates GGML_BACKEND_API being a visibility attribute
#                       at all — without it the macro is a bare `extern`
#   GGML_BACKEND_BUILD  selects dllexport over dllimport within that
#
# Missing GGML_BACKEND_SHARED is a silent failure: the library builds, loads,
# and the loader then reports "failed to find ggml_backend_init". Nothing warns
# that an entry point was declared but never exported.
target_compile_definitions(moeserv PRIVATE
    GGML_BACKEND_DL
    GGML_BACKEND_SHARED
    GGML_BACKEND_BUILD)

# ggml-backend-impl.h is an internal header: the vtables a backend implements
# are not part of the public API. Every out-of-tree backend needs it, and using
# it is what couples this project to a ggml version — a compile error after a
# ggml update is the intended signal, and is why the plan records the commit
# these interfaces were read at.
target_include_directories(moeserv PRIVATE
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/ggml/src")

# ggml-base carries ggml_backend_buffer_init, the tensor helpers and the type
# traits. `ggml` carries the backend *registry* — ggml_backend_init_by_type and
# friends live in ggml-backend-reg.cpp, which is in that target and not in the
# base one.
#
# Not ggml-cpu, deliberately: the CPU kernels are obtained at runtime by asking
# the registry for a CPU backend, so this library contains no second copy of
# them. Both links here are *import* libraries, so at load time they resolve to
# the ggml the host llama.cpp already has — which is the point. A statically
# embedded registry would be a second registry, and we would register into the
# one nobody reads. See README.md on keeping stray ggml DLLs away from the
# output directory for the same reason.
target_link_libraries(moeserv PRIVATE ggml-base ggml)

set_target_properties(moeserv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    PREFIX "")

# moe-probe — a standalone kernel benchmark, deliberately not part of the
# backend. It links ggml normally because it needs the kernels in its own
# process; the library never does, since it borrows the host's.
#
# --custom runs the raw-Vulkan 2-pass MXFP4 prototype: its own instance and
# pipelines, ggml only as the CPU reference. The GLSL is compiled at build time
# by the SDK's glslc and loaded from bin/shaders/ next to the executable.
find_package(Vulkan REQUIRED)
set(MOE_SHADERS mxfp4_pass1 mxfp4_pass2 tp_mid tp_final)
set(MOE_SPV_FILES)
foreach(sh ${MOE_SHADERS})
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/bin/shaders/${sh}.spv"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin/shaders"
        COMMAND Vulkan::glslc -O -o "${CMAKE_BINARY_DIR}/bin/shaders/${sh}.spv"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${sh}.comp"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${sh}.comp"
        COMMENT "glslc ${sh}.comp")
    list(APPEND MOE_SPV_FILES "${CMAKE_BINARY_DIR}/bin/shaders/${sh}.spv")
endforeach()
add_custom_target(moe-shaders DEPENDS ${MOE_SPV_FILES})

add_executable(moe-probe src/moe_probe.cpp src/moe_probe_custom.cpp)
add_dependencies(moe-probe moe-shaders)
target_include_directories(moe-probe PRIVATE "${LLAMA_CPP_DIR}/ggml/include")
target_link_libraries(moe-probe PRIVATE ggml-base ggml Vulkan::Vulkan)
set_target_properties(moe-probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
