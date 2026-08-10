#pragma once

// Build fingerprint: the set of facts that decide whether two runs can be
// expected to produce the same bits. One definition, three consumers —
// `--version`, the provenance sidecar the gate script writes, and the protocol
// handshake, where it crosses the wire so the client can see what the backend
// actually is.
//
// The list is not arbitrary. Every entry here can move logits, or gates
// something that does:
//
//   compiler   FP contraction is a compiler choice, not an ISA one — a build
//              that contracts a*b+c into an FMA sums differently from one that
//              does not, at the same optimization level on the same chip.
//   ggml       kernel changes are the whole point of tracking upstream.
//   blas       engaged at batch >= 32 via op offload; changes prefill numerics.
//   llamafile  sgemm tiling participates in the summation order.
//   threads    ggml partitions matmul work by thread count, so -t 16 and -t 32
//              are different numbers — this one is measured and unambiguous.
//   git rev    catches source changes the four above cannot.
//
// Format is `key=value`, one per line — greppable, diffable, and parsed by
// gate.py without a dependency. Not JSON: this has to be assembled in C++ with
// no library, and a flat key-value list is the shape where that is not a
// mistake.
//
// NOTE the git rev is captured at *configure* time, not build time, so it goes
// stale if you commit and rebuild without reconfiguring. That is deliberate —
// a build-time git invocation is a custom target and a generated header for a
// field the gate script cross-checks anyway: gate.py records the live rev
// separately, and a disagreement between the two means exactly "this binary
// predates HEAD", which is worth knowing rather than hiding.

#include "cpu_topology.h"

#include "ggml.h"

#include <cstdint>
#include <map>
#include <string>
#include <thread>

#ifndef NANO_GIT_REV
#   define NANO_GIT_REV "unknown"
#endif
#ifndef NANO_COMPILER_ID
#   define NANO_COMPILER_ID "unknown"
#endif
#ifndef NANO_COMPILER_VERSION
#   define NANO_COMPILER_VERSION "0"
#endif
#ifndef NANO_BUILD_BLAS
#   define NANO_BUILD_BLAS 0
#endif
#ifndef NANO_BUILD_LLAMAFILE
#   define NANO_BUILD_LLAMAFILE 0
#endif
#ifndef NANO_BUILD_VULKAN
#   define NANO_BUILD_VULKAN 0
#endif
#if defined(NANO_EXPERT_TRACE)
#   define NANO_BUILD_TRACE 1
#else
#   define NANO_BUILD_TRACE 0
#endif

// Build-time half: identical for every run of one binary.
static std::string nano_build_info() {
    std::string s;
    s += "git_rev=";   s += NANO_GIT_REV;        s += "\n";
    s += "compiler=";  s += NANO_COMPILER_ID;
    s += " ";          s += NANO_COMPILER_VERSION; s += "\n";
    s += "ggml_version="; s += ggml_version();   s += "\n";
    s += "ggml_commit=";  s += ggml_commit();    s += "\n";
    s += "blas=";      s += (NANO_BUILD_BLAS      ? "1" : "0"); s += "\n";
    s += "llamafile="; s += (NANO_BUILD_LLAMAFILE ? "1" : "0"); s += "\n";
    s += "vulkan=";    s += (NANO_BUILD_VULKAN    ? "1" : "0"); s += "\n";
    s += "trace=";     s += (NANO_BUILD_TRACE     ? "1" : "0"); s += "\n";
    return s;
}

// One-line form for startup banners. Same macros as above, so the two can
// disagree about which fields they show but never about a value.
static std::string nano_build_line() {
    std::string s = "build ";
    s += NANO_GIT_REV;
    s += " | "; s += NANO_COMPILER_ID; s += " "; s += NANO_COMPILER_VERSION;
    s += " | ggml "; s += ggml_version(); s += " ("; s += ggml_commit(); s += ")";
    s += " | blas=";     s += (NANO_BUILD_BLAS      ? "1" : "0");
    s += " llamafile=";  s += (NANO_BUILD_LLAMAFILE ? "1" : "0");
    s += " vulkan=";     s += (NANO_BUILD_VULKAN    ? "1" : "0");
    s += " trace=";      s += (NANO_BUILD_TRACE     ? "1" : "0");
    return s;
}

// Run-time half: what this particular invocation was configured with. Core
// counts are informational (they explain a thread count that was defaulted),
// n_threads is not — it changes the logits.
static std::string nano_run_info(int n_threads) {
    std::string s;
    s += "n_threads="  + std::to_string(n_threads) + "\n";
    s += "cores_phys=" + std::to_string(physical_core_count()) + "\n";
    s += "cores_log="  + std::to_string((int) std::thread::hardware_concurrency()) + "\n";
    return s;
}

// Parse the same format back. Unknown keys are kept: a newer peer may send
// fields this build has never heard of, and dropping them would make the
// handshake log less useful than the wire already was.
static std::map<std::string, std::string> nano_kv_parse(const std::string & text) {
    std::map<std::string, std::string> kv;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(pos, nl - pos);
        const size_t eq = line.find('=');
        if (eq != std::string::npos) kv[line.substr(0, eq)] = line.substr(eq + 1);
        pos = nl + 1;
    }
    return kv;
}

// Fields whose disagreement voids bit-exactness without making a run invalid.
// Kept next to the producer so adding a field to nano_build_info() and
// forgetting to compare it is a one-file mistake rather than a two-file one.
// Deliberately absent: git_rev (running a different revision is the whole
// point of a comparison) and trace (a -DNANO_EXPERT_TRACE build was measured
// byte-identical to a plain one).
// `vulkan` is here for a slightly different reason than the rest: it does not
// change the *client's* numerics at all, since the client is always CPU-only.
// It voids bit-exactness because a Vulkan-enabled backend evaluates experts in
// different arithmetic, and a strict client should refuse that pairing rather
// than silently byte-compare against a golden set it cannot match. Increment 2
// deliberately turns `gate.py rpc --strict` into a refusal, which is the signal
// to use the KL gate instead.
static const char * const NANO_REPRO_KEYS[] = {
    "compiler", "ggml_commit", "blas", "llamafile", "vulkan", "n_threads",
    "model_first", "model_bytes", "model_shards",
};

// Model identity, cheaply. Basename rather than full path, because the same
// model sits somewhere else on the other machine and that is not a problem.
//
// Total bytes across every shard, not the first shard's: shard 1 of this model
// is 9.4 MB of metadata and shards 2-14 are ~49 GB each, so a first-shard size
// says almost nothing about which quantization is loaded — which is precisely
// the thing the structural hparams cannot distinguish, since Q4_K and Q6_K of
// one model share every one of them.
static std::string nano_model_info(const std::string & path, uint64_t total_bytes,
                                   uint32_t n_shards) {
    const size_t slash = path.find_last_of("/\\");
    std::string s = "model_first=" + (slash == std::string::npos ? path : path.substr(slash + 1)) + "\n";
    s += "model_bytes="  + std::to_string(total_bytes) + "\n";
    s += "model_shards=" + std::to_string(n_shards) + "\n";
    return s;
}
