#pragma once

// Where a chunk of evaluation actually goes, split into the six things every
// model's `eval_chunk` does. Generic: nothing here knows what a model is.
//
// Why it exists. `OPTIMIZATION.md` has carried an unresolved gap for a while —
// nano-glm local at 3.15 t/s against llama.cpp's 4.04 with matched kernels — and
// a named suspect it could not test: deepseek4 rebuilds its ~6000-node graph
// every chunk, which is once per decode token, where glm-dsa caches by
// (n_tokens, n_kv). That is a question about host-side work versus the forward
// pass, and no amount of end-to-end tok/s can answer it, because both are inside
// the same number.
//
// Always on, and deliberately, for the reason `moe_stats` gives: six clock reads
// per chunk against a forward pass that touches 150 GiB is free, and a number
// you only collect when you remember to ask for it is a number you do not have
// when something looks wrong. There is no flag to turn it on and none to turn it
// off.
//
// What the phases mean, and where the two models differ:
//
//   build    graph construction — `ggml_init` plus the model's `build_graph`.
//            For deepseek4 this also covers the host-side plans (`ds4_plan_comp`,
//            `ds4_plan_raw_mask`), which are index bookkeeping rather than ggml.
//            glm-dsa only pays this when (n_tokens, n_kv) changes; deepseek4
//            pays it every chunk. **The difference between those two columns is
//            the measurement this file was written for.**
//   alloc    `ggml_gallocr` / `ggml_backend_sched_alloc_graph`. deepseek4
//            constructs and destroys its allocator per chunk, so this covers the
//            compute-buffer allocation too.
//   input    host->backend uploads, and the host-side buffers they need built
//            (the attention mask, position vectors).
//   compute  `ggml_backend_graph_compute`. The forward pass, and the only phase
//            that should be large.
//   read     logits read-back.
//   free     teardown of anything the chunk allocated. Zero where nothing is.
//
// Read the *share*, not the absolute: a run pages 150 GiB in during its first
// chunks, so the first prefill is not comparable to a warm decode step and the
// per-phase percentages are what survive that.
//
// **Prefill and decode are counted separately**, and that is not a nicety. A
// 111-token prefill and a 1-token decode step differ by more than a factor in
// every phase, and one prompt-shaped run is ~78% prefill by wall time, so a
// combined average is dominated by the chunk you are usually not asking about.
// Aggregating them hid a 37 ms/chunk allocator cost that was ~3% of a run and
// ~8% of a decode step, and it made "the client trunk costs the same with and
// without the split" impossible to check — the check needs decode alone, and
// when it was finally done the arithmetic came out negative, which is how the
// loopback thread-contention question surfaced at all.

#include <chrono>
#include <cstdint>
#include <cstdio>

struct nano_phase_stats {
    uint64_t n_chunks = 0;
    uint64_t n_tokens = 0;

    uint64_t build_us   = 0;
    uint64_t alloc_us   = 0;
    uint64_t input_us   = 0;
    uint64_t compute_us = 0;
    uint64_t read_us    = 0;
    uint64_t free_us    = 0;

    uint64_t total_us() const {
        return build_us + alloc_us + input_us + compute_us + read_us + free_us;
    }
};

// Lap timer: `lap()` returns microseconds since construction or since the
// previous lap, so a sequence of phases reads as one `+= T.lap()` per phase with
// no bookkeeping between them and no gaps unaccounted for.
struct nano_phase_timer {
    std::chrono::steady_clock::time_point t;

    nano_phase_timer() : t(std::chrono::steady_clock::now()) {}

    uint64_t lap() {
        const auto now = std::chrono::steady_clock::now();
        const auto d   = std::chrono::duration_cast<std::chrono::microseconds>(now - t).count();
        t = now;
        return (uint64_t) d;
    }
};

// Prefill and decode kept apart. A chunk of one token is a decode step;
// anything wider is a prefill chunk. That rule misfiles a one-token prompt,
// which is not a case anyone measures.
struct nano_phase_split {
    nano_phase_stats prefill;
    nano_phase_stats decode;

    nano_phase_stats & bucket(int64_t n_tok) { return n_tok == 1 ? decode : prefill; }
};

// One line per phase, absolute and as a share, for one bucket.
static void nano_phase_report_one(FILE * f, const char * tag, const char * what,
                                  const nano_phase_stats & p) {
    const uint64_t tot = p.total_us();
    if (!p.n_chunks || !tot) return;

    fprintf(f, "%s: %s phases: %llu chunks / %llu tokens, %.2fs accounted, "
               "%.1f ms/chunk\n",
            tag, what, (unsigned long long) p.n_chunks, (unsigned long long) p.n_tokens,
            tot / 1e6, tot / 1e3 / (double) p.n_chunks);

    struct { const char * name; uint64_t us; } rows[] = {
        { "build",   p.build_us   },
        { "alloc",   p.alloc_us   },
        { "input",   p.input_us   },
        { "compute", p.compute_us },
        { "read",    p.read_us    },
        { "free",    p.free_us    },
    };
    for (const auto & r : rows) {
        fprintf(f, "%s:   %-8s %8.3fs  %5.1f%%  %8.2f ms/chunk\n",
                tag, r.name, r.us / 1e6, 100.0 * (double) r.us / (double) tot,
                r.us / 1e3 / (double) p.n_chunks);
    }
}

static void nano_phase_report(FILE * f, const char * tag, const nano_phase_split & s) {
    nano_phase_report_one(f, tag, "prefill", s.prefill);
    nano_phase_report_one(f, tag, "decode ", s.decode);
}
