#pragma once

// `tape` — write down what graph_compute was handed, so it can be replayed
// without llama.cpp and without a model.
//
// The format records the split *generically*: an ordered node list with ops,
// shapes, types, op_params and source indices, plus the data of every tensor
// that has no producer inside the split (the weights, the activations, the ids,
// the router weights) and of the output. Replay rebuilds the graph from that.
//
// Generic rather than "write out w_up, w_gate, w_down, x, ids" on purpose. A
// capture shaped around the five ops we expect would stop describing the graph
// the moment the graph changed, and a replay built on it would keep passing —
// which is exactly how `../nano-glm` lost a SwiGLU clamp for four commits
// between two hand-synced copies of one op sequence.
//
// Expert weights are ~137 GiB and identical on every call, so tensor data is
// content-addressed: each blob is written once to blobs/<hash>.bin and records
// reference it. A 24-token run then costs megabytes rather than terabytes.
//
// Enabled by MOESERV_CAPTURE=<dir>. Off by default and costing one getenv.

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <direct.h>
#  define MOE_MKDIR(p) _mkdir(p)
#else
#  include <sys/stat.h>
#  define MOE_MKDIR(p) mkdir(p, 0755)
#endif

// v2 adds the thread count to each record and captures the data of every
// tensor this split does not produce, rather than only the op-NONE ones. Both
// are replay's requirements; see moe_tape_write.
#define MOE_TAPE_MAGIC "MOETAPE2"

// FNV-1a over the bytes. Not cryptographic — this only has to distinguish the
// handful of distinct weight blobs in one run from each other, and a collision
// would show up immediately as a replay mismatch rather than silently.
static inline uint64_t moe_tape_hash(const void * data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

// Budgets, because the first capture written without them was 140 GB.
//
// Content addressing dedupes a weight across *calls*, but each layer's experts
// are genuinely different and llama.cpp hands `mul_mat_id` the whole
// 256-expert tensor — ~1.07 GB, times three per layer, times 43 layers. So
// "capture the inputs" degenerates into "capture the model", which is the one
// thing this was built to avoid.
//
// A budget is the blunt fix and it is not the real one: see PLAN.md on slicing
// a `mul_mat_id`'s weights down to the experts its ids actually name, which is
// ~80 MB per decode record instead of 3.2 GB and is a faithful reduction rather
// than a truncation. Until that exists, stop early and say so, because a
// capture that silently stopped being written is worse than a small one.
struct moe_tape {
    std::string dir;
    FILE *      idx = nullptr;      // records.bin
    uint32_t    n_calls = 0;        // splits seen, whether recorded or not
    uint32_t    n_records = 0;
    uint64_t    blob_bytes = 0;
    uint32_t    max_records = 8;
    uint64_t    max_bytes = 4ull << 30;
    bool        stopped = false;
    std::map<uint64_t, bool> blobs;  // blob hashes already written
    std::map<uint64_t, bool> shapes; // split shapes already recorded

    bool active() const { return idx != nullptr && !stopped; }
};

static inline void moe_tape_w32(FILE * f, uint32_t v) { fwrite(&v, 4, 1, f); }
static inline void moe_tape_w64(FILE * f, uint64_t v) { fwrite(&v, 8, 1, f); }
static inline void moe_tape_wstr(FILE * f, const char * s) {
    const uint32_t n = (uint32_t) strlen(s);
    moe_tape_w32(f, n);
    if (n) fwrite(s, 1, n, f);
}

// One blob per distinct content. Returns the hash; writes the file only the
// first time it is seen.
static inline uint64_t moe_tape_blob(moe_tape & T, const void * data, size_t n) {
    const uint64_t h = moe_tape_hash(data, n);
    if (T.blobs.find(h) == T.blobs.end()) {
        char path[512];
        snprintf(path, sizeof(path), "%s/blobs/%016llx.bin", T.dir.c_str(), (unsigned long long) h);
        FILE * f = fopen(path, "wb");
        if (f) { fwrite(data, 1, n, f); fclose(f); }
        T.blobs[h] = true;
        T.blob_bytes += n;
    }
    return h;
}

static inline void moe_tape_open(moe_tape & T, const char * dir) {
    T.dir = dir;
    MOE_MKDIR(dir);
    std::string blobs = T.dir + "/blobs";
    MOE_MKDIR(blobs.c_str());

    std::string idx = T.dir + "/records.bin";
    T.idx = fopen(idx.c_str(), "wb");
    if (!T.idx) {
        fprintf(stderr, "MoE: cannot open %s for capture\n", idx.c_str());
        return;
    }
    if (const char * v = getenv("MOESERV_CAPTURE_MAX_RECORDS")) T.max_records = (uint32_t) atoi(v);
    if (const char * v = getenv("MOESERV_CAPTURE_MAX_MB"))      T.max_bytes = (uint64_t) atoll(v) << 20;

    fwrite(MOE_TAPE_MAGIC, 1, 8, T.idx);
    moe_tape_w32(T.idx, 0);   // record count, rewritten after every record
    fprintf(stderr, "MoE: capturing to %s (one record per distinct split shape, "
                    "max %u records, %llu MB)\n",
            dir, T.max_records, (unsigned long long) (T.max_bytes >> 20));
}

static inline void moe_tape_close(moe_tape & T) {
    if (!T.idx) return;
    fseek(T.idx, 8, SEEK_SET);
    moe_tape_w32(T.idx, T.n_records);
    fclose(T.idx);
    T.idx = nullptr;
    fprintf(stderr, "MoE: capture closed, %u records, %zu blobs\n",
            T.n_records, T.blobs.size());
}

// Write one split. Called after compute, so the output tensor holds its result.
//
// A tensor's data is written when nothing inside this graph produces it (it is
// an input: weight, activation, ids, router weights) or when nothing inside
// this graph consumes it (it is an output the rest of the model reads).
// Everything between is reproducible by replaying the ops and would only bloat
// the capture.
//
// `n_threads` is part of the record because it is part of the *arithmetic*:
// ggml partitions a matmul by thread count, so the summation order and
// therefore the rounding change with it (repo CLAUDE.md), and llama.cpp uses a
// different count for prefill than for decode. A replay that guessed would be
// correct and not bit-identical, which is the worst of the two outcomes because
// it looks like a kernel difference.
static inline void moe_tape_write(moe_tape & T, ggml_cgraph * gf, int n_threads) {
    if (!T.idx || !gf) return;

    const int n = ggml_graph_n_nodes(gf);
    if (n == 0) return;

    const uint32_t call = T.n_calls++;

    // One record per distinct split *shape*, rather than the first N splits.
    //
    // The layers of one model build the same graph with different weights, so
    // "the first two records" is two prefill layers and never a decode one —
    // and prefill and decode differ in both batch shape and thread count, which
    // are the two things a replay has to get right. Keying on the op-and-shape
    // list gets exactly one of each, in a run of any length, and then costs
    // nothing for the remaining hundreds of calls.
    uint64_t key = 1469598103934665603ull;
    auto mix = [&key](uint64_t v) { key ^= v; key *= 1099511628211ull; };
    mix((uint64_t) n);
    for (int i = 0; i < n; i++) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        mix((uint64_t) t->op);
        for (int d = 0; d < GGML_MAX_DIMS; d++) mix((uint64_t) t->ne[d]);
    }
    if (T.shapes.find(key) != T.shapes.end()) return;
    T.shapes[key] = true;

    // Index every tensor reachable from the nodes, in a stable order: nodes
    // first (so a src index can refer to a node), then the leaves they pull in.
    std::vector<ggml_tensor *> ts;
    std::map<const ggml_tensor *, int> idx;
    auto add = [&](ggml_tensor * t) -> int {
        if (!t) return -1;
        auto it = idx.find(t);
        if (it != idx.end()) return it->second;
        const int i = (int) ts.size();
        ts.push_back(t);
        idx[t] = i;
        return i;
    };
    for (int i = 0; i < n; i++) add(ggml_graph_node(gf, i));
    // srcs and view sources may be leaves; add them after the nodes so node
    // indices stay equal to their graph position.
    for (size_t i = 0; i < ts.size(); i++) {
        ggml_tensor * t = ts[i];
        for (int s = 0; s < GGML_MAX_SRC; s++) add(t->src[s]);
        add(t->view_src);
    }

    const int n_nodes = n;
    const int n_all   = (int) ts.size();

    // The call ordinal, not the record ordinal: with shape dedup those differ,
    // and the ordinal is the only thing that says which split this was.
    moe_tape_w32(T.idx, call);
    moe_tape_w32(T.idx, (uint32_t) n_nodes);
    moe_tape_w32(T.idx, (uint32_t) n_all);
    moe_tape_w32(T.idx, (uint32_t) n_threads);

    // A node is terminal if nothing else in this split consumes it. Those are
    // the values the rest of the model reads, so those are what a replay has to
    // match. The first version of this captured only the *last* node, which for
    // this block is one of six views of the experts tensor — a replay built on
    // that would have compared a sixth of the result and passed.
    std::vector<bool> consumed(n_all, false);
    for (int i = 0; i < n_all; i++) {
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (ts[i]->src[s]) consumed[idx[ts[i]->src[s]]] = true;
        }
        if (ts[i]->view_src) consumed[idx[ts[i]->view_src]] = true;
    }

    for (int i = 0; i < n_all; i++) {
        ggml_tensor * t = ts[i];
        moe_tape_wstr(T.idx, ggml_get_name(t));
        // The op *name*, not just its number: the enum is internal and shifts
        // between ggml versions, and a reader that decodes 23 as MUL_MAT_ID
        // when it now means something else is a reader that lies quietly.
        moe_tape_wstr(T.idx, ggml_op_name(t->op));
        moe_tape_w32(T.idx, (uint32_t) t->type);
        moe_tape_w32(T.idx, (uint32_t) t->op);
        for (int d = 0; d < GGML_MAX_DIMS; d++) moe_tape_w64(T.idx, (uint64_t) t->ne[d]);
        for (int d = 0; d < GGML_MAX_DIMS; d++) moe_tape_w64(T.idx, (uint64_t) t->nb[d]);
        fwrite(t->op_params, 1, GGML_MAX_OP_PARAMS, T.idx);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const int si = t->src[s] ? idx[t->src[s]] : -1;
            moe_tape_w32(T.idx, (uint32_t) si);
        }
        const int vi = t->view_src ? idx[t->view_src] : -1;
        moe_tape_w32(T.idx, (uint32_t) vi);
        moe_tape_w64(T.idx, (uint64_t) t->view_offs);

        // Data for graph inputs (nothing here produces them) and for terminal
        // nodes (nothing here consumes them). Everything between is reproduced
        // by replaying the ops and would only bloat the capture.
        //
        // "Input" is `i >= n_nodes` — not in this graph's node list — rather
        // than `op == NONE`. Neither model's expert block hands us one, but a
        // src that is a *view* of something computed outside the split is a
        // legal graph, and under the narrower rule its data would be silently
        // omitted and replay would compute from an uninitialised tensor. Its
        // recorded `nb` may be strided; the byte span from `data` is still
        // exactly what the op reads, so writing that reproduces it.
        const bool is_input  = (i >= n_nodes);
        const bool is_output = (i < n_nodes) && !consumed[i];
        const bool want      = (is_input || is_output) && t->data != nullptr;
        moe_tape_w32(T.idx, want ? 1u : 0u);
        if (want) {
            const size_t nb = ggml_nbytes(t);
            moe_tape_w64(T.idx, (uint64_t) nb);
            moe_tape_w64(T.idx, moe_tape_blob(T, t->data, nb));
        }
    }
    T.n_records++;

    fprintf(stderr, "MoE: recorded call %u (%d nodes, %d threads), %.2f GB so far\n",
            call, n_nodes, n_threads, T.blob_bytes / 1073741824.0);

    if (T.n_records >= T.max_records || T.blob_bytes >= T.max_bytes) {
        T.stopped = true;
        fprintf(stderr, "MoE: capture stopped after %u records, %.1f GB "
                        "(%s limit; raise with MOESERV_CAPTURE_MAX_RECORDS / _MAX_MB)\n",
                T.n_records, T.blob_bytes / 1073741824.0,
                T.n_records >= T.max_records ? "record" : "size");
    }
}
