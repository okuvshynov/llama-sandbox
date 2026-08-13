// moe-probe — what does a die sustain on *our* mul_mat_id, by weight type?
//
//   moe-probe --device Vulkan0 --type f32,f16,q4_0,q8_0,mxfp4
//   moe-probe --device Vulkan0 --m 2048 --k 4096 --tokens 8
//
// Two questions in one tool, both of which this project has answered wrongly by
// inference before:
//
//   The ceiling. The expert block is 0.63 FLOP/byte at batch 1 — pure
//   streaming — so the only number that matters is how fast this device can
//   read scattered expert matrices at all. f32 and f16 answer that: they have
//   no unpacking cost, so whatever they reach is the memory system's practical
//   limit for this access pattern, and every quantized format is measured
//   against it rather than against a datasheet's 1024 GB/s.
//
//   The format. `test-backend-ops` puts q4_0 20x ahead of mxfp4 at *its*
//   shapes; a matched pair of real models put it 1.3-1.5x ahead at ours. These
//   kernels swing 30x with shape (q6_K: 10 GB/s at m=768, 306 at m=1792), so a
//   number measured anywhere but our own shapes predicts nothing.
//
// Deliberately standalone and deliberately synthetic: random weights, one
// mul_mat_id, no model, no llama.cpp. It measures a kernel, not a system, and
// it cannot be used to argue anything about correctness — synthetic weights
// never trigger the SwiGLU clamp, which is exactly what ../nano-glm lost across
// 240 of 256 experts for four commits. `gate.py` remains the verdict.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct probe_args {
    std::string device = "Vulkan0";
    std::string types  = "f32,f16,q4_0,q8_0,mxfp4";
    int64_t m = 2048, k = 4096;      // one expert is [k, m]; deepseek4's gate/up
    int64_t n_expert = 32;           // only n_used are read, so this only sizes the buffer
    int64_t n_used = 6;
    int64_t n_tokens = 1;
    int     reps = 20;
};

static ggml_type type_from_name(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && strcmp(n, s) == 0) return (ggml_type) t;
    }
    return GGML_TYPE_COUNT;
}

static std::vector<std::string> split_commas(const std::string & s) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= s.size()) {
        const size_t b = s.find(',', a);
        out.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return out;
}

static bool run_one(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                    const probe_args & A, ggml_type type) {
    // Skip a type this build cannot quantize into rather than aborting: the
    // point is to compare the ones that exist.
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && !ggml_is_quantized(type)) return false;
    if (A.k % ggml_blck_size(type) != 0) return false;

    const size_t overhead = 8 * ggml_tensor_overhead() + ggml_graph_overhead() + (1 << 20);
    ggml_init_params ip = { overhead, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    // as  [k, m, n_expert] — the expert bank
    // b   [k, n_used, n_tokens] — the activations, one copy per used expert
    // ids [n_used, n_tokens]
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, type, A.k, A.m, A.n_expert);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, A.k, A.n_used, A.n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, A.n_used, A.n_tokens);
    ggml_tensor * dst = ggml_mul_mat_id(ctx, as, b, ids);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { ggml_free(ctx); return false; }

    // Random data. Values are irrelevant to timing but must not be denormal or
    // NaN, which can cost real cycles on some hardware.
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> src(A.k * A.m);
    for (auto & v : src) v = dist(rng);
    {
        std::vector<uint8_t> row(ggml_nbytes(as) / A.n_expert);
        if (type == GGML_TYPE_F32) {
            memcpy(row.data(), src.data(), row.size());
        } else {
            ggml_quantize_chunk(type, src.data(), row.data(), 0, A.m, A.k, nullptr);
        }
        for (int64_t e = 0; e < A.n_expert; e++) {
            ggml_backend_tensor_set(as, row.data(), e * row.size(), row.size());
        }
    }
    {
        std::vector<float> bv(A.k * A.n_used * A.n_tokens);
        for (auto & v : bv) v = dist(rng);
        ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(float));

        // Distinct experts, so the run reads n_used separate matrices — the
        // scattered access the real router produces, not one matrix n_used times.
        std::vector<int32_t> iv(A.n_used * A.n_tokens);
        for (size_t i = 0; i < iv.size(); i++) iv[i] = (int32_t) (i % A.n_expert);
        ggml_backend_tensor_set(ids, iv.data(), 0, iv.size() * sizeof(int32_t));
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc); ggml_backend_buffer_free(buf); ggml_free(ctx);
        return false;
    }

    // Bytes the kernel must read: one whole [k, m] matrix per distinct expert.
    // Weights dominate everything else by three orders of magnitude, so the
    // activations and ids are not counted.
    const int64_t n_distinct = std::min<int64_t>(A.n_used * A.n_tokens, A.n_expert);
    const double bytes = (double) n_distinct * A.m * ggml_row_size(type, A.k);

    ggml_backend_graph_compute(backend, gf);       // warm-up: pipeline creation
    ggml_backend_synchronize(backend);

    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < A.reps; r++) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - t0).count() / A.reps;

    printf("  %-7s %8.2f MB %10.1f us %9.1f GB/s\n",
           ggml_type_name(type), bytes / 1e6, us, bytes / (us * 1e-6) / 1e9);

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    probe_args A;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if      (a == "--device") A.device   = next();
        else if (a == "--type")   A.types    = next();
        else if (a == "--m")      A.m        = atoll(next());
        else if (a == "--k")      A.k        = atoll(next());
        else if (a == "--experts")A.n_expert = atoll(next());
        else if (a == "--used")   A.n_used   = atoll(next());
        else if (a == "--tokens") A.n_tokens = atoll(next());
        else if (a == "--reps")   A.reps     = atoi(next());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }

    ggml_backend_load_all();

    ggml_backend_t backend = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (A.device == ggml_backend_dev_name(d)) { backend = ggml_backend_dev_init(d, nullptr); break; }
    }
    if (!backend) {
        fprintf(stderr, "no device named %s. available:", A.device.c_str());
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
        }
        fprintf(stderr, "\n");
        return 2;
    }

    printf("%s: mul_mat_id [k=%lld, m=%lld, experts=%lld], %lld used, %lld token(s), %d reps\n",
           A.device.c_str(), (long long) A.k, (long long) A.m, (long long) A.n_expert,
           (long long) A.n_used, (long long) A.n_tokens, A.reps);

    for (const std::string & t : split_commas(A.types)) {
        const ggml_type ty = type_from_name(t.c_str());
        if (ty == GGML_TYPE_COUNT) { printf("  %-7s unknown type\n", t.c_str()); continue; }
        if (!run_one(backend, ggml_backend_get_default_buffer_type(backend), A, ty)) {
            printf("  %-7s unsupported here\n", t.c_str());
        }
    }

    ggml_backend_free(backend);
    return 0;
}
