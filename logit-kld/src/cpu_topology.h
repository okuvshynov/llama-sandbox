#pragma once

// Physical core count, NOT std::thread::hardware_concurrency() (which counts SMT
// siblings: 32 on the 16-core Xeon W-3245 this repo runs on). Two reasons this
// matters, and the second is the load-bearing one:
//
//  - SMT buys little, and for prefill its sign is platform-dependent. Stock
//    llama-bench, GLM-5.2 UD-Q6_K, same Mac Pro 7,1, 16 vs 32 threads, 5 reps:
//
//                     generation (tg32)      prefill (pp128)
//      macOS/clang    2.00 -> 2.23  (+11%)   7.13 -> 6.43  (-10%)
//      Windows/MSVC   1.58 -> 1.84  (+16%)   6.24 -> 7.26  (+16%)
//
//    Decode is weight-bandwidth-bound, so doubling threads adds ~10-16% at
//    best; macOS prefill actively REGRESSES with SMT, making physical cores
//    the fastest prefill setting there. Don't generalise one platform's
//    prefill behaviour to the other. Short prefills are too noisy to read
//    anything into (Windows pp64 at 16 threads: 3.56 +/- 0.95 over 5 reps; an
//    earlier 2-rep sample suggested a 2x SMT gain that 5 reps did not
//    support). So the default costs ~16% at worst, and nothing on macOS
//    prefill — speed is not why it is pinned.
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
