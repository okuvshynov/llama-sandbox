// State a fatal-signal handler can print (README steps 93, 96: a rare abort at exit). Written
// with plain stores, read by the handler in ds_completion.cpp without locks or allocation.
#pragma once

#include <atomic>

// where the process is: "run", then each step of the shutdown (string literals only)
inline std::atomic<const char*> g_fatal_phase{"run"};
// the expert loader's loads per die: taken from the queue / copies and unpack finished
constexpr int kFatalDies = 4;
inline std::atomic<long> g_fatal_loads_started[kFatalDies] = {};
inline std::atomic<long> g_fatal_loads_done[kFatalDies] = {};

inline void fatal_phase(const char* p) { g_fatal_phase.store(p, std::memory_order_relaxed); }
