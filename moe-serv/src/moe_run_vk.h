#pragma once

// `dies` increment 3: run a placed layer's split on its die.
//
// llama.cpp hands us a cgraph whose tensors live in host memory. A Vulkan
// backend cannot read those, so the split is rebuilt against device-resident
// twins: the expert weights come from the mirror, the handful of inputs are
// uploaded per call, the intermediates are allocated in VRAM, and the terminals
// are read back into the host tensors the rest of the model will read.
//
// **Rebuilt generically from the cgraph, never from a hand-written copy of the
// ops we expect.** The graph is currently MUL_MAT_ID, CLAMP, MUL_MAT_ID, CLAMP,
// GLU, MUL_MAT_ID, MUL and six VIEWs, and writing that sequence out here would
// be a second definition of the thing under test — which is exactly how
// ../nano-glm ran 240 of 256 experts without a SwiGLU clamp for four commits,
// between two hand-synced copies of one op sequence. Nothing below names an op.
//
// Two things are deliberately checked rather than assumed:
//
//   Every node is offered to the device before anything is built. Vulkan has no
//   kernel for some ops this architecture uses, and a model whose block we
//   cannot run must fall back to the CPU and be slow, not wrong.
//
//   The allocator is kept across calls. Rebuilding it per graph cost ../nano-glm
//   37 ms/chunk in `free` alone, plus a soft-fault bill on the next chunk that
//   its phase timer attributed to compute.

#include "moe_mirror.h"
#include "moe_place.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <map>
#include <vector>

struct moe_vk {
    std::vector<ggml_backend_t> backends;   // one per device, created lazily
    std::vector<ggml_gallocr_t> gallocs;    // kept across calls, see above
    std::vector<bool>           refused;    // device cannot run this block
    std::vector<bool>           ran;        // has computed at least one split
    // Splits actually computed here, by device, plus the ones that went to the
    // CPU. Reported at teardown: "the backend was engaged" and "the dies did
    // most of the work" are different claims, and a benchmark that cannot tell
    // them apart will eventually quote a number for the wrong configuration.
    std::vector<uint64_t>       n_split;
    uint64_t                    n_cpu = 0;

    void resize(size_t n) {
        backends.resize(n, nullptr);
        gallocs.resize(n, nullptr);
        refused.resize(n, false);
        ran.resize(n, false);
        n_split.resize(n, 0);
    }
};

static inline void moe_vk_free(moe_vk & V) {
    for (ggml_gallocr_t g : V.gallocs) if (g) ggml_gallocr_free(g);
    for (ggml_backend_t b : V.backends) if (b) ggml_backend_free(b);
    V.gallocs.clear();
    V.backends.clear();
    V.refused.clear();
}

// Which die should run this split, or -1 for the CPU.
//
// Decided from the weights the graph actually references rather than from a
// counter: a split whose three expert matmuls are not all mirrored on one device
// has no business on that device, and saying so here is cheaper than discovering
// it half-way through building a graph.
static inline int moe_vk_device_for(const moe_placement & P, const moe_mirror & M, ggml_cgraph * gf) {
    int dev = -1;
    int n_weights = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        const ggml_tensor * n = ggml_graph_node(gf, i);
        if (n->op != GGML_OP_MUL_MAT_ID) continue;
        n_weights++;
        const int layer = moe_place_layer_of(ggml_get_name(n->src[0]));
        if (layer < 0 || layer >= (int) P.layer_dev.size()) return -1;
        const int d = P.layer_dev[layer];
        if (d < 0) return -1;
        if (M.copy.find(n->src[0]) == M.copy.end()) return -1;
        if (dev >= 0 && d != dev) return -1;
        dev = d;
    }
    return n_weights > 0 ? dev : -1;
}

// Tokens per dispatch.
//
// `ggml_vk_use_mul_mat_vec_id` takes the vector path only when the ids tensor's
// token count is <= 8 (ggml-vulkan.cpp:10607); one token more and it takes the
// general path. Measured on the 4-layer stub, that boundary is worth +21.9% at
// 8 tokens and -60.2% at 16 — the same cliff ../nano-glm found from the other
// side. So the block is issued 8 tokens at a time rather than in one piece.
#define MOE_VK_CHUNK 8

// Which dimension of `t` counts tokens, or -1 for "none", or -2 for "ambiguous".
//
// Ambiguity is a real possibility rather than paranoia — a tensor whose expert
// count or feature width happens to equal the token count would be sliced along
// the wrong axis and produce fluent nonsense — so it is detected and refuses the
// chunking rather than guessing.
static inline int moe_vk_token_dim(const ggml_tensor * t, int64_t n_tok) {
    int found = -1;
    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        if (t->ne[d] != n_tok) continue;
        if (found >= 0) return -2;
        found = d;
    }
    return found;
}

// Rebuild and run. Returns false when the split was not run here, in which case
// the caller must fall back to the CPU — every early return below is a case
// where being slow is right and guessing would be wrong.
static inline bool moe_vk_compute(moe_placement & P, moe_mirror & M, moe_vk & V,
                                  int dev, ggml_cgraph * gf, const char * tag) {
    V.resize(P.devs.size());
    if (dev < 0 || dev >= (int) P.devs.size() || V.refused[dev]) return false;

    if (!V.backends[dev]) {
        V.backends[dev] = ggml_backend_dev_init(P.devs[dev], nullptr);
        if (!V.backends[dev]) {
            fprintf(stderr, "%s: %s failed to initialise; its layers stay on the CPU\n",
                    tag, P.dev_name[dev].c_str());
            V.refused[dev] = true;
            return false;
        }
    }

    // Ask before building. An op this device cannot run makes the whole split
    // the CPU's, permanently — falling back per call would hide it.
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        if (!ggml_backend_dev_supports_op(P.devs[dev], ggml_graph_node(gf, i))) {
            fprintf(stderr, "%s: %s has no kernel for %s; this block stays on the CPU\n",
                    tag, P.dev_name[dev].c_str(), ggml_op_name(ggml_graph_node(gf, i)->op));
            V.refused[dev] = true;
            return false;
        }
    }

    const int n_nodes = ggml_graph_n_nodes(gf);

    // Index every tensor the graph reaches, nodes first so a source index can
    // refer to a node, then the leaves they pull in.
    std::vector<ggml_tensor *> host;
    std::map<const ggml_tensor *, int> idx;
    auto add = [&](ggml_tensor * t) -> int {
        if (!t) return -1;
        auto it = idx.find(t);
        if (it != idx.end()) return it->second;
        const int i = (int) host.size();
        host.push_back(t);
        idx[t] = i;
        return i;
    };
    for (int i = 0; i < n_nodes; i++) add(ggml_graph_node(gf, i));
    for (size_t i = 0; i < host.size(); i++) {
        for (int s = 0; s < GGML_MAX_SRC; s++) add(host[i]->src[s]);
        add(host[i]->view_src);
    }
    const int n_all = (int) host.size();

    // A node is terminal when nothing else here consumes it; those are the
    // values the rest of the model reads and the only ones worth copying back.
    // The first version of the capture format used "the last node" for this and
    // would have compared one sixth of the result — this block's terminals are
    // six views of one tensor.
    std::vector<bool> consumed(n_all, false);
    for (int i = 0; i < n_all; i++) {
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (host[i]->src[s]) consumed[idx[host[i]->src[s]]] = true;
        }
        if (host[i]->view_src) consumed[idx[host[i]->view_src]] = true;
    }

    // How many tokens this split covers, and where each tensor keeps them.
    // Read from the ids tensor rather than guessed: it is the tensor whose
    // second dimension the Vulkan backend actually tests.
    int64_t n_tok = 0;
    for (int i = 0; i < n_nodes && n_tok == 0; i++) {
        const ggml_tensor * n = ggml_graph_node(gf, i);
        if (n->op == GGML_OP_MUL_MAT_ID && n->src[2]) n_tok = n->src[2]->ne[1];
    }
    std::vector<int> tok_dim(n_all, -1);
    int64_t chunk = n_tok > 0 ? n_tok : 1;
    if (n_tok > MOE_VK_CHUNK) {
        chunk = MOE_VK_CHUNK;
        for (int i = 0; i < n_all; i++) {
            if (M.copy.find(host[i]) != M.copy.end()) continue;   // a weight, never sliced
            tok_dim[i] = moe_vk_token_dim(host[i], n_tok);
            if (tok_dim[i] == -2) {                                // cannot tell: do not guess
                static bool said = false;
                if (!said) {
                    said = true;
                    fprintf(stderr, "%s: %s has two dimensions of %lld; running unchunked\n",
                            tag, ggml_get_name(host[i]), (long long) n_tok);
                }
                chunk = n_tok;
                break;
            }
        }
    }

    ggml_init_params ip = {
        /* .mem_size   = */ (size_t) (n_all + 16) * ggml_tensor_overhead()
                            + ggml_graph_overhead_custom(n_all + 16, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(P.devs[dev]);
    if (!V.gallocs[dev]) V.gallocs[dev] = ggml_gallocr_new(buft);

    // One pass per chunk of tokens. A chunk twin keeps the host tensor's
    // *strides* and shrinks only the token dimension, so its byte span is
    // exactly the host tensor's span starting at t0*nb[d] — which is what keeps
    // the upload and the read-back single contiguous copies instead of a
    // per-token scatter.
    for (int64_t t0 = 0; t0 < (n_tok > 0 ? n_tok : 1); t0 += chunk) {
        const int64_t k = (n_tok > 0 && t0 + chunk > n_tok) ? n_tok - t0 : chunk;

        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return false;

        // Twins. Built by writing the source tensor's fields into a bare tensor
        // rather than by calling ggml_mul_mat_id / ggml_clamp / ..., because the
        // op constructors are where a rebuild would start disagreeing with the
        // graph it is meant to be running.
        std::vector<ggml_tensor *> twin(n_all, nullptr);
        std::vector<int> uploads;
        for (int i = 0; i < n_all; i++) {
            ggml_tensor * h = host[i];
            auto mirrored = M.copy.find(h);
            if (mirrored != M.copy.end()) {
                twin[i] = mirrored->second;      // already in VRAM, nothing to do
                continue;
            }
            int64_t ne[GGML_MAX_DIMS];
            for (int d = 0; d < GGML_MAX_DIMS; d++) ne[d] = h->ne[d];
            if (tok_dim[i] >= 0) ne[tok_dim[i]] = k;
            ggml_tensor * t = ggml_new_tensor(ctx, h->type, GGML_MAX_DIMS, ne);
            if (!t) { ggml_free(ctx); return false; }
            ggml_set_name(t, ggml_get_name(h));
            for (int d = 0; d < GGML_MAX_DIMS; d++) t->nb[d] = h->nb[d];
            memcpy(t->op_params, h->op_params, GGML_MAX_OP_PARAMS);
            if (i < n_nodes) {
                t->op = h->op;
            } else if (h->data) {
                uploads.push_back(i);            // an input: activations, ids, router weights
            }
            twin[i] = t;
        }
        for (int i = 0; i < n_nodes; i++) {
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                twin[i]->src[s] = host[i]->src[s] ? twin[idx[host[i]->src[s]]] : nullptr;
            }
            if (host[i]->view_src) {
                twin[i]->view_src  = twin[idx[host[i]->view_src]];
                twin[i]->view_offs = host[i]->view_offs;
            }
        }

        // Expanded one node at a time in the reference's own order: each node's
        // sources are already in the hash set, so expand appends exactly that
        // node.
        ggml_cgraph * dg = ggml_new_graph_custom(ctx, (size_t) n_all + 16, false);
        for (int i = 0; i < n_nodes; i++) ggml_build_forward_expand(dg, twin[i]);

        if (!ggml_gallocr_alloc_graph(V.gallocs[dev], dg)) {
            fprintf(stderr, "%s: %s could not allocate the block's graph; back to the CPU\n",
                    tag, P.dev_name[dev].c_str());
            V.refused[dev] = true;
            ggml_free(ctx);
            return false;
        }

        for (int i : uploads) {
            const size_t off = tok_dim[i] >= 0 ? (size_t) t0 * host[i]->nb[tok_dim[i]] : 0;
            ggml_backend_tensor_set(twin[i], (const char *) host[i]->data + off,
                                    0, ggml_nbytes(twin[i]));
        }

        const enum ggml_status st = ggml_backend_graph_compute(V.backends[dev], dg);
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: %s returned status %d; back to the CPU\n",
                    tag, P.dev_name[dev].c_str(), (int) st);
            V.refused[dev] = true;
            ggml_free(ctx);
            return false;
        }

        for (int i = 0; i < n_nodes; i++) {
            if (consumed[i] || !host[i]->data) continue;
            const size_t off = tok_dim[i] >= 0 ? (size_t) t0 * host[i]->nb[tok_dim[i]] : 0;
            ggml_backend_tensor_get(twin[i], (char *) host[i]->data + off,
                                    0, ggml_nbytes(twin[i]));
        }

        ggml_free(ctx);
        if (n_tok == 0) break;
    }

    // Announced once per device, because "the backend was engaged" and "the
    // device computed something" are different claims and this project has
    // already shipped four checks that proved neither.
    if (!V.ran[dev]) {
        V.ran[dev] = true;
        fprintf(stderr, "%s: %s computed a split (%d nodes, %lld tokens in chunks of %lld)\n",
                tag, P.dev_name[dev].c_str(), n_nodes,
                (long long) n_tok, (long long) chunk);
    }
    V.n_split[dev]++;
    return true;
}

static inline void moe_vk_report(const moe_placement & P, const moe_vk & V, const char * tag) {
    uint64_t gpu = 0;
    for (uint64_t n : V.n_split) gpu += n;
    if (gpu == 0 && V.n_cpu == 0) return;
    fprintf(stderr, "%s: splits computed — %llu on device(s), %llu on the CPU\n",
            tag, (unsigned long long) gpu, (unsigned long long) V.n_cpu);
    for (size_t d = 0; d < V.n_split.size() && d < P.dev_name.size(); d++) {
        if (V.n_split[d]) {
            fprintf(stderr, "%s:   %-9s %llu\n", tag, P.dev_name[d].c_str(),
                    (unsigned long long) V.n_split[d]);
        }
    }
}
