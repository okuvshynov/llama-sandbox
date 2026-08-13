#pragma once

// `dies` increment 2: put a copy of the placed layers' expert weights in VRAM.
//
// A *mirror*, not a move. The weights stay in our host buffer and a second copy
// goes to the die, which costs 137 GiB of host RAM plus up to 115 GiB of VRAM on
// a machine that has both. Two reasons it is worth that:
//
//   `is_host` stays true. It is what lets llama.cpp's CPU backend read our
//   tensors in place, so every op we do not claim remains free, and it is what
//   keeps the correctness gate comparing the same two things it always has.
//
//   llama.cpp allocates one buffer per buffer type, so a device-memory buffer
//   type could not express "84% of layers on the GPU, 16% on the CPU" at all.
//   Placement has to be ours to decide per layer, and a mirror is how.
//
// Uploaded lazily at the first graph_compute rather than at load: that is the
// first moment every weight has actually been written, and a process that
// registers the backend and never computes (llama-bench listing devices, the
// -ot probe) should not move 115 GiB for nothing.

#include "moe_place.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <map>
#include <vector>

struct moe_mirror {
    // Per device: the context holding the copies, the buffer backing them, and
    // the host tensors still to upload.
    std::vector<ggml_context *>        ctxs;
    std::vector<ggml_backend_buffer_t> bufs;
    std::vector<std::vector<ggml_tensor *>> pending;

    // Host tensor -> its VRAM copy. Empty until upload, and the only thing
    // graph_compute needs to look at.
    std::map<const ggml_tensor *, ggml_tensor *> copy;

    bool done = false;
};

// Called from init_tensor once placement has decided. Records the tensor; no
// VRAM is touched here, because the data does not exist yet.
static inline void moe_mirror_note(moe_mirror & M, int dev, ggml_tensor * t) {
    if (dev < 0) return;
    if ((int) M.pending.size() <= dev) M.pending.resize(dev + 1);
    M.pending[dev].push_back(t);
}

// Undo a device's placement and send its layers back to the CPU. Called when
// allocation fails: the reserve is an estimate, and being wrong about it must
// degrade rather than abort — a run that falls back is slow and correct, a run
// that aborts after a ten-minute load is neither.
static inline void moe_mirror_fallback(moe_placement & P, moe_mirror & M, int dev, const char * tag) {
    fprintf(stderr, "%s: %s could not hold its share; those layers fall back to the CPU\n",
            tag, P.dev_name[dev].c_str());
    for (size_t l = 0; l < P.layer_dev.size(); l++) {
        if (P.layer_dev[l] == dev) P.layer_dev[l] = -1;
    }
    for (ggml_tensor * t : M.pending[dev]) M.copy.erase(t);
    M.pending[dev].clear();
}

static inline void moe_mirror_upload(moe_placement & P, moe_mirror & M, const char * tag) {
    if (M.done) return;
    M.done = true;
    if (M.pending.empty()) return;

    const auto t0 = std::chrono::steady_clock::now();
    M.ctxs.resize(P.devs.size(), nullptr);
    M.bufs.resize(P.devs.size(), nullptr);

    size_t total = 0;
    for (size_t d = 0; d < M.pending.size() && d < P.devs.size(); d++) {
        if (M.pending[d].empty()) continue;

        ggml_init_params ip = {
            /* .mem_size   = */ (M.pending[d].size() + 1) * ggml_tensor_overhead(),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        M.ctxs[d] = ggml_init(ip);
        if (!M.ctxs[d]) { moe_mirror_fallback(P, M, (int) d, tag); continue; }

        std::vector<ggml_tensor *> copies;
        copies.reserve(M.pending[d].size());
        size_t bytes = 0;
        for (ggml_tensor * h : M.pending[d]) {
            ggml_tensor * c = ggml_new_tensor(M.ctxs[d], h->type, GGML_MAX_DIMS, h->ne);
            if (!c) { copies.clear(); break; }
            ggml_set_name(c, ggml_get_name(h));
            copies.push_back(c);
            bytes += ggml_nbytes(h);
        }
        if (copies.size() != M.pending[d].size()) { moe_mirror_fallback(P, M, (int) d, tag); continue; }

        M.bufs[d] = ggml_backend_alloc_ctx_tensors_from_buft(M.ctxs[d], ggml_backend_dev_buffer_type(P.devs[d]));
        if (!M.bufs[d]) { moe_mirror_fallback(P, M, (int) d, tag); continue; }

        for (size_t i = 0; i < copies.size(); i++) {
            ggml_tensor * h = M.pending[d][i];
            // Straight from our host buffer, which is ordinary memory — this is
            // the whole reason the mirror is cheap to fill.
            ggml_backend_tensor_set(copies[i], h->data, 0, ggml_nbytes(h));
            M.copy[h] = copies[i];
        }
        total += bytes;
        fprintf(stderr, "%s:   %-9s uploaded %6.2f GiB in %zu tensors\n",
                tag, P.dev_name[d].c_str(), bytes / 1073741824.0, copies.size());
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "%s: mirror %.2f GiB in %.1f s (%.2f GiB/s)\n",
            tag, total / 1073741824.0, secs, secs > 0 ? total / 1073741824.0 / secs : 0.0);
}

// Full reset, not just a release: a process that loads a second model must
// re-probe and re-upload rather than answer from a map whose keys are now
// dangling pointers into a freed model.
static inline void moe_mirror_free(moe_mirror & M) {
    for (ggml_backend_buffer_t b : M.bufs) if (b) ggml_backend_buffer_free(b);
    for (ggml_context * c : M.ctxs) if (c) ggml_free(c);
    M.bufs.clear();
    M.ctxs.clear();
    M.copy.clear();
    M.pending.clear();
    M.done = false;
}
