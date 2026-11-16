// Shared by the runtime's source files (rt.cpp, rt_trace.cpp): not part of its interface.
#pragma once

#include <hip/hip_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    hipError_t e_ = (x);                                                                           \
    if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); \
  } while (0)

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
