#pragma once

// `dies` increment 1: decide which layer's experts go to which Vulkan die.
//
// Whole layers, packed in device order, remainder on the CPU. Three measurements
// from ../nano-glm decide that shape and they are worth restating, because each
// one rules out a policy that looks better on paper:
//
//   One die is as fast as four (20.2 / 20.6 / 20.0 s, sd 0.1-0.2) — the dies are
//   bound by something fixed per dispatch rather than by throughput, so *adding
//   dies buys VRAM capacity, not speed*, and a policy should minimise dispatches.
//   Whole layers means one dispatch per layer; striping a layer across four dies
//   means ~3.3, for 7pp more residency worth ~1.4% of decode.
//
//   Routing skew does not transfer across prompts — a placement built from other
//   prompts catches 28.2% of selections against 23.1% for random — so a
//   hot-expert policy is ~5pp for a lot of machinery, and is not built.
//
//   Time is linear in the slots left on the CPU, so resident fraction is the
//   only quantity worth maximising.
//
// The plan is computed from what the devices report, printed, and then obeyed.
// Increment 1 prints it and does nothing else: a placement that is only printed
// must not move a logit, and the gate is what says it did not.

#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// -1 means "stays on the CPU"; -2 means "not seen yet". Other values index the
// dev* vectors, which are parallel.
struct moe_placement {
    std::vector<int> layer_dev;      // by layer index, sized as layers are seen
    std::vector<size_t> dev_free;    // VRAM still unclaimed, by device
    std::vector<std::string> dev_name;
    std::vector<ggml_backend_dev_t> devs;
    size_t reserve_per_dev = 0;      // held back for activations and compute
    bool   planned = false;
};

// Ask the host registry for every GPU device. Nothing is allocated here: this
// only reads what the drivers report, so it is safe to call before deciding
// whether the run will use them at all.
static inline void moe_place_probe(moe_placement & P) {
    if (P.planned) return;
    P.planned = true;

    // Held back per die for the activations, the compute buffer and driver
    // overhead. 1 GiB is generous next to a 3.19 GiB layer; being wrong in this
    // direction costs one layer of residency, being wrong in the other aborts a
    // load after several minutes.
    P.reserve_per_dev = 1024ull << 20;
    if (const char * v = getenv("MOESERV_RESERVE_MB")) {
        P.reserve_per_dev = (size_t) atoll(v) << 20;
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        size_t free_ = 0, total = 0;
        ggml_backend_dev_memory(d, &free_, &total);
        P.devs.push_back(d);
        P.dev_name.push_back(ggml_backend_dev_name(d));
        P.dev_free.push_back(free_ > P.reserve_per_dev ? free_ - P.reserve_per_dev : 0);
    }
}

// Layer index from a tensor name like "blk.7.ffn_gate_exps.weight". Returns -1
// when the name does not carry one, which is a reason to leave the tensor on the
// CPU rather than to guess: a tensor we cannot place is a tensor we cannot
// account for.
static inline int moe_place_layer_of(const char * name) {
    if (!name || strncmp(name, "blk.", 4) != 0) return -1;
    char * end = nullptr;
    const long v = strtol(name + 4, &end, 10);
    if (end == name + 4 || *end != '.') return -1;
    return (int) v;
}

// Claim `bytes` for `layer` on the first die with room. Called once per expert
// tensor as the loader writes it, so a layer's three tensors land together —
// they are equal in size and are always requested in the same order, so the
// first one to find room means all three fit.
static inline int moe_place_assign(moe_placement & P, int layer, size_t bytes) {
    if (layer < 0) return -1;
    if ((int) P.layer_dev.size() <= layer) P.layer_dev.resize(layer + 1, -2);  // -2: unseen

    if (P.layer_dev[layer] >= 0) {
        // Already placed; charge this tensor to the same die.
        const int d = P.layer_dev[layer];
        P.dev_free[d] = P.dev_free[d] > bytes ? P.dev_free[d] - bytes : 0;
        return d;
    }
    if (P.layer_dev[layer] == -1) return -1;   // already decided: CPU

    for (size_t d = 0; d < P.dev_free.size(); d++) {
        // Room for all three of the layer's tensors, not just this one. Packing
        // in device order keeps a layer whole, which is the entire point of the
        // policy — half a layer on a die would need a combine.
        if (P.dev_free[d] >= 3 * bytes) {
            P.dev_free[d] -= bytes;
            P.layer_dev[layer] = (int) d;
            return (int) d;
        }
    }
    P.layer_dev[layer] = -1;
    return -1;
}

static inline void moe_place_report(const moe_placement & P, const char * tag) {
    if (P.dev_name.empty()) {
        fprintf(stderr, "%s: no GPU devices; every layer stays on the CPU\n", tag);
        return;
    }
    int on_gpu = 0, on_cpu = 0;
    for (int d : P.layer_dev) {
        if (d >= 0) on_gpu++;
        else if (d == -1) on_cpu++;
    }
    fprintf(stderr, "%s: placement — %d of %d expert layers on %zu device(s), %d on CPU\n",
            tag, on_gpu, on_gpu + on_cpu, P.dev_name.size(), on_cpu);
    for (size_t d = 0; d < P.dev_name.size(); d++) {
        std::string layers;
        for (size_t l = 0; l < P.layer_dev.size(); l++) {
            if (P.layer_dev[l] == (int) d) {
                layers += (layers.empty() ? "" : ",") + std::to_string(l);
            }
        }
        fprintf(stderr, "%s:   %-9s %6.2f GiB spare  layers %s\n", tag,
                P.dev_name[d].c_str(), P.dev_free[d] / 1073741824.0,
                layers.empty() ? "(none)" : layers.c_str());
    }
}
