#pragma once

// Physical core count, NOT std::thread::hardware_concurrency() (which counts SMT
// siblings: 32 on the 16-core Xeon W-3245 this repo runs on). Two reasons this
// matters, and the second is the load-bearing one:
//
//  - SMT buys little for this workload, in either regime. Stock llama-bench on
//    GLM-5.2 UD-Q6_K, same machine, 5 reps: generation (tg32) 1.58 t/s at 16
//    threads vs 1.84 at 32, and prefill (pp128) 6.24 vs 7.26 — about +16% for
//    twice the threads, decode being weight-bandwidth-bound and prefill not
//    scaling much past the physical cores either. This harness's own runs agree
//    (nano-glm 270 decodes: 1.27 vs 1.34). Short prefills are too noisy to read
//    anything into (pp64 at 16 threads: 3.56 ± 0.95 over 5 reps; an earlier
//    2-rep sample suggested a 2x SMT gain that 5 reps did not support).
//    So the default costs ~16% at most — speed is not why it is pinned.
//  - ggml partitions matmul work by thread count, so the number of threads
//    CHANGES THE NUMERICS: verified on GLM-5.2, -t 16 and -t 32 produce
//    different logits, while repeated runs at a fixed count are bit-identical.
//    Every tool in this toolchain therefore has to agree on the default, or an
//    A/A comparison between two of them silently measures thread-count noise
//    instead of the thing under test.
//
// Shared by logit-kld's collect/rescore and ../nano-glm so one definition
// governs all of them.

#include <thread>

#if defined(_WIN32)
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <vector>
#elif defined(__APPLE__)
#   include <sys/sysctl.h>
#else
#   include <algorithm>
#   include <cstdio>
#   include <string>
#   include <vector>
#endif

inline int physical_core_count() {
    const int logical = (int) std::thread::hardware_concurrency();
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return logical;
    std::vector<char> buf(len);
    auto * first = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *) buf.data();
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &len)) return logical;
    // Entries are variable-length and MUST be walked by info->Size. Bounding the
    // loop by sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) instead drops the
    // last entry — the struct is a union whose size is dominated by
    // GROUP_RELATIONSHIP and exceeds a real per-core record (reported 15 of 16).
    int cores = 0;
    for (DWORD off = 0; off < len; ) {
        auto * info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *) (buf.data() + off);
        if (info->Size == 0 || off + info->Size > len) break;
        if (info->Relationship == RelationProcessorCore) cores++;
        off += info->Size;
    }
    return cores > 0 ? cores : logical;
#elif defined(__APPLE__)
    int cores = 0;
    size_t sz = sizeof(cores);
    if (sysctlbyname("hw.physicalcpu", &cores, &sz, nullptr, 0) == 0 && cores > 0) return cores;
    return logical;
#else
    // Linux: each cpu lists its SMT siblings, so the number of distinct sibling
    // sets is the core count. Falls back to logical if topology is unreadable.
    std::vector<std::string> seen;
    for (int cpu = 0; cpu < logical; cpu++) {
        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        FILE * f = fopen(path, "r");
        if (!f) return logical;
        char line[256] = {0};
        const bool ok = fgets(line, sizeof(line), f) != nullptr;
        fclose(f);
        if (!ok) return logical;
        std::string s(line);
        if (std::find(seen.begin(), seen.end(), s) == seen.end()) seen.push_back(s);
    }
    return seen.empty() ? logical : (int) seen.size();
#endif
}
