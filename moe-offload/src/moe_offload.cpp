// moe-offload — cross-platform measurement of what it costs to run MoE routed
// experts on a Vulkan GPU while the trunk stays on the CPU.
//
// This exists to answer one question that decides an OS: for the dispatch
// pattern nano-glm's distributed plan needs (per layer: upload an activation,
// run top-K experts, read K rows back, combine on CPU), how much does the
// driver charge for the round trip — and does that differ between the AMD
// Vulkan driver on Windows and MoltenVK-on-Metal on macOS? Same Mac Pro 7,1,
// same Vega dies, so a difference is the software stack.
//
// It is deliberately NOT a matmul benchmark. The expert kernel is a plain
// int8 workgroup-per-row matmul (8 bpw, close to the real Q6_K's 6.64, and
// without nibble-unpack bugs to chase). What is measured precisely is the
// per-phase breakdown: upload / record / submit / wait / download, plus a
// null-kernel floor, because "dispatch is slow" has different fixes depending
// on whether the cost lands in recording, submission, or the fence.
//
// Shapes default to GLM-5.2: n_embd 6144, expert FFN 2048, top-8 of 256.

#include "vk_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

static double us_since(clk::time_point t0) {
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

struct Stats {
    double p50 = 0, p90 = 0, mean = 0, min = 0, max = 0;
};

static Stats summarize(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.p50  = v[v.size() / 2];
    s.p90  = v[(size_t)(v.size() * 0.9) < v.size() ? (size_t)(v.size() * 0.9) : v.size() - 1];
    s.min  = v.front();
    s.max  = v.back();
    double sum = 0; for (double x : v) sum += x;
    s.mean = sum / v.size();
    return s;
}

struct Config {
    uint32_t device   = 0;
    uint32_t d_embd   = 6144;   // GLM-5.2
    uint32_t d_ffn    = 2048;   // expert_feed_forward_length
    uint32_t n_expert = 32;     // experts held resident on the GPU
    uint32_t top_k    = 8;
    uint32_t n_layers = 78;     // for the per-token extrapolation only
    uint32_t iters    = 50;
    uint32_t threads  = 0;      // CPU shared expert; 0 = hardware_concurrency
    bool     list     = false;
    bool     verify   = true;
};

static void usage() {
    std::printf(
        "Usage: moe-offload [options]\n"
        "  --device N      Vulkan device index (default 0)\n"
        "  --list          list devices and exit\n"
        "  --experts N     experts resident on the GPU (default 32)\n"
        "  --topk N        experts selected per token (default 8)\n"
        "  --embd N        model dim (default 6144)\n"
        "  --ffn N         expert FFN dim (default 2048)\n"
        "  --layers N      MoE layers, for per-token extrapolation (default 78)\n"
        "  --iters N       timed iterations per mode (default 50)\n"
        "  --threads N     CPU threads for the shared expert (default: all)\n"
        "  --no-verify     skip the CPU cross-check of GPU expert output\n");
}

// ---------------------------------------------------------------------------
// host-side int8 weights

// Deterministic filler: identical bytes on both platforms so the CPU/GPU
// cross-check and any cross-machine comparison see the same numbers.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 1) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    int8_t   i8()   { return (int8_t)(next() & 0xFF); }
    // Cast to signed BEFORE subtracting: `next() % 2001 - 1000` is unsigned
    // arithmetic, so any value below 1000 wraps to ~4.29e9 and the whole
    // harness ends up multiplying numbers of order 1e6.
    float    unit() { return (float)((int32_t)(next() % 2001) - 1000) / 1000.0f; }
};

// y[o] = scale[o] * sum_k W[o][k] * x[k]   (CPU reference / shared expert)
static void matmul_i8_cpu(const int8_t * W, const float * scale, const float * x,
                          float * y, uint32_t O, uint32_t K,
                          uint32_t nthread, uint32_t o_begin, uint32_t o_end) {
    (void) nthread;
    for (uint32_t o = o_begin; o < o_end; ++o) {
        const int8_t * row = W + (size_t) o * K;
        float acc = 0.0f;
        for (uint32_t k = 0; k < K; ++k) acc += (float) row[k] * x[k];
        y[o] = acc * scale[o];
    }
}

static void parallel_matmul_i8(const int8_t * W, const float * scale, const float * x,
                               float * y, uint32_t O, uint32_t K, uint32_t nthread) {
    if (nthread <= 1) { matmul_i8_cpu(W, scale, x, y, O, K, 1, 0, O); return; }
    std::vector<std::thread> pool;
    uint32_t chunk = (O + nthread - 1) / nthread;
    for (uint32_t t = 0; t < nthread; ++t) {
        uint32_t b = t * chunk, e = std::min(O, b + chunk);
        if (b >= e) break;
        pool.emplace_back([&, b, e] { matmul_i8_cpu(W, scale, x, y, O, K, 1, b, e); });
    }
    for (auto & th : pool) th.join();
}

int main(int argc, char ** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> uint32_t { return (uint32_t) std::stoul(argv[++i]); };
        if      (a == "--device"  && i + 1 < argc) cfg.device   = val();
        else if (a == "--experts" && i + 1 < argc) cfg.n_expert = val();
        else if (a == "--topk"    && i + 1 < argc) cfg.top_k    = val();
        else if (a == "--embd"    && i + 1 < argc) cfg.d_embd   = val();
        else if (a == "--ffn"     && i + 1 < argc) cfg.d_ffn    = val();
        else if (a == "--layers"  && i + 1 < argc) cfg.n_layers = val();
        else if (a == "--iters"   && i + 1 < argc) cfg.iters    = val();
        else if (a == "--threads" && i + 1 < argc) cfg.threads  = val();
        else if (a == "--list")       cfg.list   = true;
        else if (a == "--no-verify")  cfg.verify = false;
        else { usage(); return a == "--help" ? 0 : 1; }
    }
    if (cfg.threads == 0) cfg.threads = std::max(1u, std::thread::hardware_concurrency());

    if (cfg.list) {
        auto devs = vkc::Context::listDevices();
        std::printf("%zu Vulkan device(s):\n", devs.size());
        for (size_t i = 0; i < devs.size(); ++i) {
            const auto & d = devs[i];
            std::printf("  %zu: %s | driver %s | api %u.%u.%u | maxAlloc %.2f GiB | subgroup %u\n",
                        i, d.name.c_str(), d.driver.c_str(),
                        VK_VERSION_MAJOR(d.apiVersion), VK_VERSION_MINOR(d.apiVersion),
                        VK_VERSION_PATCH(d.apiVersion),
                        (double) d.maxAllocationSize / (1024.0 * 1024.0 * 1024.0), d.subgroupSize);
        }
        return 0;
    }

    if (cfg.top_k > cfg.n_expert) { std::fprintf(stderr, "topk > experts\n"); return 1; }
    if (cfg.d_embd % 4 || cfg.d_ffn % 4) { std::fprintf(stderr, "embd and ffn must be multiples of 4\n"); return 1; }

    const uint32_t D = cfg.d_embd, I = cfg.d_ffn, E = cfg.n_expert, K = cfg.top_k;
    const size_t   mat_elems = (size_t) I * D;             // up/gate: I x D; down: D x I
    const size_t   per_expert_bytes = 3 * mat_elems;       // int8

    vkc::Context ctx = vkc::Context::create(cfg.device);
    std::printf("device %u: %s | driver %s | api %u.%u.%u | maxAlloc %.2f GiB | subgroup %u\n",
                cfg.device, ctx.info.name.c_str(), ctx.info.driver.c_str(),
                VK_VERSION_MAJOR(ctx.info.apiVersion), VK_VERSION_MINOR(ctx.info.apiVersion),
                VK_VERSION_PATCH(ctx.info.apiVersion),
                (double) ctx.info.maxAllocationSize / (1024.0 * 1024.0 * 1024.0),
                ctx.info.subgroupSize);
    std::printf("shape: d_embd=%u d_ffn=%u experts=%u top_k=%u | %.2f GiB of expert weights "
                "(%.1f MiB/expert, 3 buffers of %.2f GiB)\n",
                D, I, E, K,
                (double)(E * per_expert_bytes) / (1024.0 * 1024.0 * 1024.0),
                (double) per_expert_bytes / (1024.0 * 1024.0),
                (double)(E * mat_elems) / (1024.0 * 1024.0 * 1024.0));

    // One buffer per projection, all E experts inside it. Splitting this way is
    // what keeps each allocation under the driver's maxMemoryAllocationSize
    // (2 GiB on the AMD/Windows driver) — a single tensor holding every
    // expert's weights would not fit there.
    vkc::Buffer b_wup   = ctx.createStorageBuffer(E * mat_elems);
    vkc::Buffer b_wgate = ctx.createStorageBuffer(E * mat_elems);
    vkc::Buffer b_wdown = ctx.createStorageBuffer(E * mat_elems);
    vkc::Buffer b_sup   = ctx.createStorageBuffer((size_t) E * I * sizeof(float));
    vkc::Buffer b_sgate = ctx.createStorageBuffer((size_t) E * I * sizeof(float));
    vkc::Buffer b_sdown = ctx.createStorageBuffer((size_t) E * D * sizeof(float));
    vkc::Buffer b_x     = ctx.createStorageBuffer((size_t) D * sizeof(float));
    vkc::Buffer b_up    = ctx.createStorageBuffer((size_t) K * I * sizeof(float));
    vkc::Buffer b_gate  = ctx.createStorageBuffer((size_t) K * I * sizeof(float));
    vkc::Buffer b_h     = ctx.createStorageBuffer((size_t) K * I * sizeof(float));
    vkc::Buffer b_y     = ctx.createStorageBuffer((size_t) K * D * sizeof(float));
    vkc::Buffer b_null  = ctx.createStorageBuffer(256);

    // Upload one expert at a time so the host never holds more than one
    // matrix; expert 0 is kept for the CPU cross-check.
    std::printf("uploading weights ...\n");
    std::vector<int8_t> mat(mat_elems);
    std::vector<float>  scl(std::max(I, D));
    std::vector<int8_t> keep_up, keep_gate, keep_down;
    std::vector<float>  keep_sup, keep_sgate, keep_sdown;
    Rng rng(12345);
    for (uint32_t e = 0; e < E; ++e) {
        for (int which = 0; which < 3; ++which) {
            const uint32_t rows = (which == 2) ? D : I;
            const uint32_t cols = (which == 2) ? I : D;
            for (size_t j = 0; j < mat_elems; ++j) mat[j] = rng.i8();
            for (uint32_t r = 0; r < rows; ++r) scl[r] = 0.0005f * rng.unit();
            const vkc::Buffer & wb = (which == 0) ? b_wup : (which == 1) ? b_wgate : b_wdown;
            const vkc::Buffer & sb = (which == 0) ? b_sup : (which == 1) ? b_sgate : b_sdown;
            ctx.upload(wb, mat.data(), (size_t) rows * cols, (VkDeviceSize) e * mat_elems);
            ctx.upload(sb, scl.data(), (size_t) rows * sizeof(float),
                       (VkDeviceSize) e * rows * sizeof(float));
            if (e == 0) {
                auto & kw = (which == 0) ? keep_up : (which == 1) ? keep_gate : keep_down;
                auto & ks = (which == 0) ? keep_sup : (which == 1) ? keep_sgate : keep_sdown;
                kw.assign(mat.begin(), mat.end());
                ks.assign(scl.begin(), scl.begin() + rows);
            }
        }
    }

    const std::string spv = MOE_SPV_DIR;
    // Separate jobs, one descriptor set each: the shader indexes a fixed
    // binding, so up/gate/down get their own bound weight buffer and the
    // expert is selected by push-constant offset.
    struct MatmulPC { uint32_t K, w_off, s_off, x_off, y_off; };
    vkc::ComputeJob job_up   = vkc::ComputeJob::create(ctx, spv + "/matmul_i8.spv", 4, sizeof(MatmulPC));
    vkc::ComputeJob job_gate = vkc::ComputeJob::create(ctx, spv + "/matmul_i8.spv", 4, sizeof(MatmulPC));
    vkc::ComputeJob job_down = vkc::ComputeJob::create(ctx, spv + "/matmul_i8.spv", 4, sizeof(MatmulPC));
    vkc::ComputeJob job_silu = vkc::ComputeJob::create(ctx, spv + "/silu_mul.spv", 3, sizeof(uint32_t));
    vkc::ComputeJob job_null = vkc::ComputeJob::create(ctx, spv + "/nullk.spv", 1, 0);

    job_up  .bind(0, b_wup);   job_up  .bind(1, b_sup);   job_up  .bind(2, b_x); job_up  .bind(3, b_up);
    job_gate.bind(0, b_wgate); job_gate.bind(1, b_sgate); job_gate.bind(2, b_x); job_gate.bind(3, b_gate);
    job_down.bind(0, b_wdown); job_down.bind(1, b_sdown); job_down.bind(2, b_h); job_down.bind(3, b_y);
    job_silu.bind(0, b_gate);  job_silu.bind(1, b_up);    job_silu.bind(2, b_h);
    job_null.bind(0, b_null);

    // Host state: activation, router weights, shared-expert weights.
    std::vector<float> x(D);
    for (uint32_t i = 0; i < D; ++i) x[i] = rng.unit();
    std::vector<int8_t> w_router((size_t) E * D);
    for (auto & v : w_router) v = rng.i8();
    std::vector<float> s_router(E, 0.0005f);
    std::vector<int8_t> w_sh_up((size_t) I * D), w_sh_gate((size_t) I * D), w_sh_down((size_t) D * I);
    for (auto & v : w_sh_up)   v = rng.i8();
    for (auto & v : w_sh_gate) v = rng.i8();
    for (auto & v : w_sh_down) v = rng.i8();
    std::vector<float> s_sh_up(I, 0.0005f), s_sh_gate(I, 0.0005f), s_sh_down(D, 0.0005f);

    std::vector<float> y_host((size_t) K * D);
    std::vector<uint32_t> sel(K);

    // -- CPU trunk pieces -----------------------------------------------------
    auto cpu_router = [&](std::vector<uint32_t> & out) {
        std::vector<float> logits(E);
        parallel_matmul_i8(w_router.data(), s_router.data(), x.data(), logits.data(), E, D, cfg.threads);
        std::vector<uint32_t> idx(E);
        for (uint32_t i = 0; i < E; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                          [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
        out.assign(idx.begin(), idx.begin() + K);
    };
    std::vector<float> sh_up(I), sh_gate(I), sh_h(I), sh_y(D);
    auto cpu_shared_expert = [&]() {
        parallel_matmul_i8(w_sh_up.data(),   s_sh_up.data(),   x.data(), sh_up.data(),   I, D, cfg.threads);
        parallel_matmul_i8(w_sh_gate.data(), s_sh_gate.data(), x.data(), sh_gate.data(), I, D, cfg.threads);
        for (uint32_t i = 0; i < I; ++i) {
            float g = sh_gate[i];
            sh_h[i] = (g / (1.0f + std::exp(-g))) * sh_up[i];
        }
        parallel_matmul_i8(w_sh_down.data(), s_sh_down.data(), sh_h.data(), sh_y.data(), D, I, cfg.threads);
    };

    // -- record one layer's expert work into the command buffer ---------------
    auto record_layer = [&](VkCommandBuffer c) {
        for (uint32_t s = 0; s < K; ++s) {
            MatmulPC pc{D, (uint32_t)((size_t) sel[s] * mat_elems / 4), sel[s] * I, 0, s * I};
            job_up.recordGroups(c, I, &pc);
            job_gate.recordGroups(c, I, &pc);
        }
        ctx.computeBarrier(c);
        uint32_t n = K * I;
        job_silu.recordGroups(c, (n + 255) / 256, &n);
        ctx.computeBarrier(c);
        for (uint32_t s = 0; s < K; ++s) {
            MatmulPC pc{I, (uint32_t)((size_t) sel[s] * mat_elems / 4), sel[s] * D, s * I, s * D};
            job_down.recordGroups(c, D, &pc);
        }
    };

    // -- correctness: GPU experts vs the CPU reference, on expert 0 -----------
    if (cfg.verify) {
        std::vector<uint32_t> saved = sel;
        sel.assign(K, 0);
        ctx.upload(b_x, x.data(), D * sizeof(float));
        VkCommandBuffer c = ctx.beginCmd();
        record_layer(c);
        ctx.endSubmitWait();
        ctx.download(b_y, y_host.data(), (size_t) K * D * sizeof(float));

        std::vector<float> r_up(I), r_gate(I), r_h(I), r_y(D);
        matmul_i8_cpu(keep_up.data(),   keep_sup.data(),   x.data(), r_up.data(),   I, D, 1, 0, I);
        matmul_i8_cpu(keep_gate.data(), keep_sgate.data(), x.data(), r_gate.data(), I, D, 1, 0, I);
        for (uint32_t i = 0; i < I; ++i) {
            float g = r_gate[i];
            r_h[i] = (g / (1.0f + std::exp(-g))) * r_up[i];
        }
        matmul_i8_cpu(keep_down.data(), keep_sdown.data(), r_h.data(), r_y.data(), D, I, 1, 0, D);

        double max_abs = 0, rms_ref = 0, nan_count = 0;
        for (uint32_t i = 0; i < D; ++i) {
            double gpu = y_host[i];
            if (std::isnan(gpu) || std::isinf(gpu)) nan_count++;
            max_abs = std::max(max_abs, std::fabs(gpu - r_y[i]));
            rms_ref += (double) r_y[i] * r_y[i];
        }
        rms_ref = std::sqrt(rms_ref / D);
        std::printf("verify: max|gpu-cpu| = %.3e  (ref rms %.3e, rel %.2e)  non-finite: %.0f\n",
                    max_abs, rms_ref, rms_ref > 0 ? max_abs / rms_ref : 0.0, nan_count);
        // The GPU tree-reduces where the CPU sums serially, so the low bits
        // differ by construction; NaN or a large relative error does not.
        if (nan_count > 0 || (rms_ref > 0 && max_abs / rms_ref > 1e-3)) {
            std::printf("verify: FAILED — GPU expert output does not match the CPU reference.\n");
            return 2;
        }
        std::printf("verify: OK\n");
        sel = saved;
    }

    // -- measurement ----------------------------------------------------------
    const uint32_t warmup = 5;
    std::vector<double> t_null, t_up, t_down_x, t_rec, t_sub, t_wait, t_layer, t_overlap, t_cpu;

    for (uint32_t it = 0; it < cfg.iters + warmup; ++it) {
        const bool timed = it >= warmup;
        cpu_router(sel);
        for (uint32_t s = 0; s < K; ++s) sel[s] %= E;

        // (1) null dispatch: the driver's submit + queue round-trip floor
        {
            auto t0 = clk::now();
            VkCommandBuffer c = ctx.beginCmd();
            job_null.recordGroups(c, 1, nullptr);
            ctx.endSubmitWait();
            if (timed) t_null.push_back(us_since(t0));
        }

        // (2) transfers alone, at the sizes the protocol actually moves
        {
            auto t0 = clk::now();
            ctx.upload(b_x, x.data(), D * sizeof(float));
            if (timed) t_up.push_back(us_since(t0));
            auto t1 = clk::now();
            ctx.download(b_y, y_host.data(), (size_t) K * D * sizeof(float));
            if (timed) t_down_x.push_back(us_since(t1));
        }

        // (3) a full layer, phase by phase, one submit
        {
            auto t_layer0 = clk::now();
            ctx.upload(b_x, x.data(), D * sizeof(float));

            auto t0 = clk::now();
            VkCommandBuffer c = ctx.beginCmd();
            record_layer(c);
            ctx.endCmd();
            if (timed) t_rec.push_back(us_since(t0));

            auto t1 = clk::now();
            ctx.submit();
            if (timed) t_sub.push_back(us_since(t1));

            auto t2 = clk::now();
            ctx.waitFence();
            if (timed) t_wait.push_back(us_since(t2));

            ctx.download(b_y, y_host.data(), (size_t) K * D * sizeof(float));
            if (timed) t_layer.push_back(us_since(t_layer0));
        }

        // (4) the same layer with the CPU shared expert between submit and
        //     wait — the moe_send/moe_recv overlap PLAN.md phase 3 wants
        {
            auto t0 = clk::now();
            ctx.upload(b_x, x.data(), D * sizeof(float));
            VkCommandBuffer c = ctx.beginCmd();
            record_layer(c);
            ctx.endCmd();
            ctx.submit();
            auto tc = clk::now();
            cpu_shared_expert();
            double cpu_us = us_since(tc);
            ctx.waitFence();
            ctx.download(b_y, y_host.data(), (size_t) K * D * sizeof(float));
            if (timed) { t_overlap.push_back(us_since(t0)); t_cpu.push_back(cpu_us); }
        }
    }

    auto row = [&](const char * name, const std::vector<double> & v) {
        Stats s = summarize(v);
        std::printf("  %-22s p50 %9.1f   p90 %9.1f   mean %9.1f   min %9.1f  us\n",
                    name, s.p50, s.p90, s.mean, s.min);
    };

    std::printf("\n-- per-layer costs (%u iterations, top_k=%u) --\n", cfg.iters, K);
    std::printf("   (upload %.1f KiB, download %.1f KiB per layer)\n",
                D * sizeof(float) / 1024.0, K * D * sizeof(float) / 1024.0);
    row("null dispatch",        t_null);
    row("upload x",             t_up);
    row("download K rows",      t_down_x);
    row("record cmdbuf",        t_rec);
    row("submit",               t_sub);
    row("wait fence",           t_wait);
    row("LAYER total",          t_layer);
    row("LAYER w/ cpu overlap", t_overlap);
    row("  cpu shared expert",  t_cpu);

    const double layer_p50   = summarize(t_layer).p50;
    const double overlap_p50 = summarize(t_overlap).p50;
    const double cpu_p50     = summarize(t_cpu).p50;
    const double null_p50    = summarize(t_null).p50;

    std::printf("\n-- extrapolation to %u MoE layers (decode, batch 1) --\n", cfg.n_layers);
    std::printf("  sequential:      %8.1f ms/token   (%.2f tok/s if nothing else cost anything)\n",
                layer_p50 * cfg.n_layers / 1000.0, 1000.0 / (layer_p50 * cfg.n_layers / 1000.0));
    std::printf("  with overlap:    %8.1f ms/token\n", overlap_p50 * cfg.n_layers / 1000.0);
    std::printf("  driver floor:    %8.1f ms/token   (null dispatch x %u layers)\n",
                null_p50 * cfg.n_layers / 1000.0, cfg.n_layers);
    const double serial = layer_p50 + cpu_p50;
    std::printf("  overlap saved:   %8.1f%% of (gpu layer + cpu shexp)\n",
                serial > 0 ? 100.0 * (serial - overlap_p50) / serial : 0.0);
    std::printf("\nNote: the driver-floor row is the part no kernel optimisation can remove.\n"
                "If it dominates, the fix is fewer submits per token (batch layers, or keep\n"
                "more of the trunk on the GPU), not a faster expert kernel.\n");

    job_up.destroy(); job_gate.destroy(); job_down.destroy(); job_silu.destroy(); job_null.destroy();
    for (vkc::Buffer * b : { &b_wup, &b_wgate, &b_wdown, &b_sup, &b_sgate, &b_sdown,
                             &b_x, &b_up, &b_gate, &b_h, &b_y, &b_null }) ctx.destroyBuffer(*b);
    ctx.destroy();
    return 0;
}
