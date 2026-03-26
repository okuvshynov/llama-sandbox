cmake_minimum_required(VERSION 3.16)

project(serv)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)

include(FetchContent)

FetchContent_Declare(
    llama.cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        b3547
)
FetchContent_MakeAvailable(llama.cpp)

add_executable(duo  duo.cpp)

target_link_libraries(duo  PRIVATE common) # from llama.cpp
target_compile_definitions(duo PRIVATE LLAMA_RPC=ON)

#configure_file(${llama.cpp_SOURCE_DIR}/ggml/src/ggml-metal.metal ggml-metal.metal COPYONLY)
#configure_file(${llama.cpp_SOURCE_DIR}/ggml/src/ggml-common.h ggml-common.h COPYONLY)

if(MSVC)
  target_compile_options(duo  PRIVATE /W4 /WX)
else()
  target_compile_options(duo  PRIVATE -Wall -Wextra -Wpedantic)
endif()

