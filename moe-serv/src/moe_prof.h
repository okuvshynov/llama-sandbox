#pragma once

// Per-split phase timing, written as CSV for something else to read.
//
// Enabled by MOESERV_PROFILE=<prefix>, off otherwise at the cost of one getenv.
// The file is <prefix>-<pid>.csv, because a benchmark runs several processes
// against one path and the interesting one would otherwise be overwritten by
// whichever finished last.
//
// Rows are accumulated in memory and written at teardown rather than fprintf'd
// as they happen: the phases being measured are tens of microseconds, and
// buffered file I/O inside the measured region would be a large fraction of
// what it is measuring.
//
// Timing both the Vulkan path and the CPU delegate matters more than timing
// either alone. The interesting decode question is not "how long does a die
// take" but "how does that compare with the 16 cores it replaced, on the same
// layer of the same model" — and with `dev = -1` for the CPU rows, one file
// answers it.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <process.h>
#  define MOE_GETPID _getpid
#else
#  include <unistd.h>
#  define MOE_GETPID getpid
#endif

struct moe_prof_row {
    uint32_t call;
    int32_t  dev;        // -1 = the CPU delegate
    int32_t  n_nodes;
    int64_t  n_tok;
    int64_t  chunk;
    int64_t  n_chunks;
    int64_t  build_us;
    int64_t  alloc_us;
    int64_t  upload_us;
    int64_t  compute_us;
    int64_t  read_us;
    int64_t  free_us;
    int64_t  total_us;
};

struct moe_prof {
    bool     checked = false;
    bool     on      = false;
    std::string prefix;
    std::vector<moe_prof_row> rows;
    uint32_t n_calls = 0;
};

// steady_clock, not high_resolution_clock: the latter is an alias for
// system_clock in some standard libraries and is therefore not monotonic.
struct moe_timer {
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    int64_t lap() {
        const auto now = std::chrono::steady_clock::now();
        const auto us  = std::chrono::duration_cast<std::chrono::microseconds>(now - t).count();
        t = now;
        return (int64_t) us;
    }
};

static inline bool moe_prof_on(moe_prof & P) {
    if (!P.checked) {
        P.checked = true;
        if (const char * p = getenv("MOESERV_PROFILE")) {
            P.prefix = p;
            P.on = true;
            P.rows.reserve(1 << 16);
        }
    }
    return P.on;
}

static inline void moe_prof_add(moe_prof & P, const moe_prof_row & r) {
    if (P.on) P.rows.push_back(r);
}

static inline void moe_prof_flush(moe_prof & P, const char * tag) {
    if (!P.on || P.rows.empty()) return;

    char path[1024];
    snprintf(path, sizeof(path), "%s-%d.csv", P.prefix.c_str(), (int) MOE_GETPID());
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "%s: cannot write %s\n", tag, path);
        P.rows.clear();
        return;
    }
    fprintf(f, "call,dev,n_nodes,n_tok,chunk,n_chunks,"
               "build_us,alloc_us,upload_us,compute_us,read_us,free_us,total_us\n");
    for (const moe_prof_row & r : P.rows) {
        fprintf(f, "%u,%d,%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld\n",
                r.call, r.dev, r.n_nodes,
                (long long) r.n_tok, (long long) r.chunk, (long long) r.n_chunks,
                (long long) r.build_us, (long long) r.alloc_us, (long long) r.upload_us,
                (long long) r.compute_us, (long long) r.read_us, (long long) r.free_us,
                (long long) r.total_us);
    }
    fclose(f);

    // A summary on stderr too, so a run can be read without opening the file.
    // Split by device because a mixed placement has both kinds of row and their
    // means are the comparison worth seeing.
    fprintf(stderr, "%s: profile -> %s (%zu rows)\n", tag, path, P.rows.size());
    for (int dev = -1; dev < 8; dev++) {
        int64_t n = 0, build = 0, alloc = 0, up = 0, comp = 0, rd = 0, fr = 0, tot = 0;
        for (const moe_prof_row & r : P.rows) {
            if (r.dev != dev) continue;
            n++; build += r.build_us; alloc += r.alloc_us; up += r.upload_us;
            comp += r.compute_us; rd += r.read_us; fr += r.free_us; tot += r.total_us;
        }
        if (!n) continue;
        fprintf(stderr, "%s:   %-8s n=%-6lld mean us: build %.1f alloc %.1f upload %.1f "
                        "compute %.1f read %.1f free %.1f | total %.1f\n",
                tag, dev < 0 ? "cpu" : ("vk" + std::to_string(dev)).c_str(),
                (long long) n, (double) build / n, (double) alloc / n, (double) up / n,
                (double) comp / n, (double) rd / n, (double) fr / n, (double) tot / n);
    }
    P.rows.clear();
}
