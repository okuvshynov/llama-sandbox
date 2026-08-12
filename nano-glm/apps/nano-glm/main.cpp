// nano-glm — the validation harness: raw token ids in, an lkldtopk v1 file out.
//
// No tokenizer, no chat template, no sampler; greedy is the stored top-1. That
// is not minimalism for its own sake — the bit-exactness contract is defined
// over a *fixed token sequence*, so anything able to change that sequence has
// to live in a different binary or every stored reference silently rots. See
// PLAN.md step 7.
//
// The engine it drives is in ../../lib: models/glm_dsa/graph.h holds the backends, KV
// cache and the op-for-op port of llama.cpp's glm-dsa trunk graph, moe_client.h
// the optional remote MoE seam. What is left here is policy — arguments, where
// the prompt comes from, the greedy loop, what gets written.
//
// models/glm_dsa/graph.h first: it reaches moe_proto.h, and winsock2.h must precede the
// windows.h that models/glm_dsa/model.h and cpu_topology.h pull in.
#include "models/glm_dsa/graph.h"

#include "cpu_topology.h"
#include "expert_trace.h"
#include "logits_file.h"
#include "models/deepseek4/eval.h"
#include "models/deepseek4/model.h"
#include "models/glm_dsa/model.h"
#include "prompt_source.h"
#include "topk_utils.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// params

struct nano_params {
    std::string model_path;      // first shard
    std::string input_bin;       // -i: lkldtopk file, prompt tokens taken from it
    std::string tokens_str;      // -T: comma-separated token ids
    std::string output_path = "nano.bin";
    int32_t     n_predict   = 256;
    int32_t     top_k       = 128;
    int32_t     n_ctx       = 4096;
    int32_t     n_batch     = 512;
    int32_t     n_threads   = (int32_t) physical_core_count();
    std::string moe_addr;        // host:port — routed experts go to moe-server
    std::string moe_log;         // JSONL of per-RPC timings
    std::string expert_log;      // per-position, per-layer selected expert ids
    bool        moe_strict = false;  // handshake: reproducibility drift is fatal
};

static bool parse_args(int argc, char ** argv, nano_params & p) {
    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        if      (!strcmp(a, "-m") && i + 1 < argc) { p.model_path  = argv[++i]; }
        else if (!strcmp(a, "-i") && i + 1 < argc) { p.input_bin   = argv[++i]; }
        else if (!strcmp(a, "-T") && i + 1 < argc) { p.tokens_str  = argv[++i]; }
        else if (!strcmp(a, "-o") && i + 1 < argc) { p.output_path = argv[++i]; }
        else if (!strcmp(a, "-n") && i + 1 < argc) { p.n_predict   = atoi(argv[++i]); }
        else if (!strcmp(a, "-k") && i + 1 < argc) { p.top_k       = atoi(argv[++i]); }
        else if (!strcmp(a, "-c") && i + 1 < argc) { p.n_ctx       = atoi(argv[++i]); }
        else if (!strcmp(a, "-b") && i + 1 < argc) { p.n_batch     = atoi(argv[++i]); }
        else if (!strcmp(a, "-t") && i + 1 < argc) { p.n_threads   = atoi(argv[++i]); }
        else if (!strcmp(a, "--moe-addr") && i + 1 < argc) { p.moe_addr = argv[++i]; }
        else if (!strcmp(a, "--moe-log")  && i + 1 < argc) { p.moe_log  = argv[++i]; }
        else if (!strcmp(a, "--expert-log") && i + 1 < argc) { p.expert_log = argv[++i]; }
        else if (!strcmp(a, "--strict"))                     { p.moe_strict = true; }
        else {
            fprintf(stderr, "nano-glm: unknown argument '%s'\n", a);
            p.model_path.clear();
            break;
        }
    }
    if (p.model_path.empty() || (p.input_bin.empty() == p.tokens_str.empty())) {
        fprintf(stderr,
            "Usage: nano-glm -m <first-shard.gguf> (-i <lkldtopk.bin> | -T <id,id,...>) [options]\n"
            "  -i <path>   take prompt token ids from an lkldtopk file (its n_prompt tokens)\n"
            "  -T <ids>    comma-separated prompt token ids\n"
            "  -o <path>   output lkldtopk file (default: nano.bin)\n"
            "  -n <int>    tokens to generate, greedy (default: 256)\n"
            "  -k <int>    top-K logits stored per position (default: 128)\n"
            "  -c <int>    context size, auto-raised to fit (default: 4096)\n"
            "  -b <int>    prompt chunk size (default: 512)\n"
            "  -t <int>    threads (default: physical cores, ignoring SMT siblings)\n"
            "  --moe-addr <host:port>  evaluate routed experts on a remote moe-server\n"
            "              instead of locally (see ../PLAN.md step 1)\n"
            "  --moe-log <path>  write per-RPC timings as JSONL\n"
            "  --strict    with --moe-addr, refuse a backend whose build differs in any\n"
            "              way that voids bit-exactness (compiler, ggml, threads, model).\n"
            "              A structurally different model is always refused; this covers\n"
            "              the reproducibility fields the gate depends on.\n"
            "  --expert-log <path>  write the routing trace (selected expert ids per\n"
            "              position per layer); requires a -DNANO_EXPERT_TRACE build\n"
            "              and the local MoE path (see lib/expert_trace.h)\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// main


int main(int argc, char ** argv) {
    // Before anything else, and without a model: the gate script reads this to
    // fill the provenance sidecar, so it must work on a bare binary.
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--version")) {
            fputs(nano_build_info().c_str(), stdout);
            fputs(nano_run_info((int) physical_core_count()).c_str(), stdout);
            return 0;
        }
    }

    nano_params params;
    if (!parse_args(argc, argv, params)) return 1;

    std::string label;
    std::vector<int32_t> prompt = load_prompt_tokens(params.input_bin, params.tokens_str,
                                                    label, "nano-glm");
    const int32_t n_prompt = (int32_t) prompt.size();

    // The four things this app needs from an architecture. Everything else it
    // does — arguments, chunking, the greedy loop, the file it writes — is
    // policy and identical for both models, so it is written once.
    //
    // This is a dispatch in one translation unit, not a framework: nothing in
    // `lib/` knows it exists, and the two graphs share not one line. The seam
    // is where lib/README.md predicted it would be if it appeared at all — the
    // *decomposition*, not the arithmetic.
    struct engine {
        uint32_t    n_vocab = 0;
        uint32_t    n_layer = 0;
        uint32_t    n_embd  = 0;
        int32_t     eos_id  = -1;
        std::string desc;
        // tokens, count, n_past, want-every-position -> logits [n_vocab * n_out]
        std::function<const float * (const int32_t *, int32_t, int32_t, bool)> eval;
    };

    const auto t_load_start = std::chrono::steady_clock::now();

    std::string arch;
    {
        ggml_context * pctx = nullptr;
        gguf_init_params pgp = { /*no_alloc =*/ true, /*ctx =*/ &pctx };
        gguf_context * pg = gguf_init_from_file(params.model_path.c_str(), pgp);
        if (!pg) NANO_ABORT("cannot read GGUF '%s'", params.model_path.c_str());
        arch = kv_str_opt(pg, "general.architecture", "?");
        gguf_free(pg);
        ggml_free(pctx);
    }
    const bool is_ds4 = arch == "deepseek4";

    nano_model M;
    ds4_model  M4;
    if (is_ds4) {
        ds4_load_model(M4, params.model_path);
    } else {
        load_model(M, params.model_path);
    }

    engine E4;
    if (is_ds4) {
        E4.n_vocab = M4.h.n_vocab;
        E4.n_layer = M4.h.n_layer;
        E4.n_embd  = M4.h.n_embd;
        E4.eos_id  = (int32_t) M4.h.eos_id;
        E4.desc    = M4.desc;
    } else {
        E4.n_vocab = M.h.n_vocab;
        E4.n_layer = M.h.n_layer;
        E4.n_embd  = M.h.n_embd;
        E4.eos_id  = (int32_t) M.h.eos_id;
        E4.desc    = M.desc;
    }

    for (int32_t t : prompt) {
        if (t < 0 || (uint32_t) t >= E4.n_vocab) {
            NANO_ABORT("prompt token %d out of vocab range [0, %u)", t, E4.n_vocab);
        }
    }

    // Connect before the graph is built: build_graph checks g_moe.active() to
    // decide whether the routed block is a local subgraph or an RPC node.
    if (!params.moe_addr.empty()) {
        const size_t colon = params.moe_addr.rfind(':');
        if (colon == std::string::npos) NANO_ABORT("--moe-addr must be host:port");
        const std::string host = params.moe_addr.substr(0, colon);
        const int         port = atoi(params.moe_addr.c_str() + colon + 1);
        if (!moe_net_init()) NANO_ABORT("socket init failed");
        g_moe.sock = moe_connect(host, (uint16_t) port);
        if (!g_moe.active()) {
            NANO_ABORT("cannot reach moe-server at %s (%s)", params.moe_addr.c_str(),
                       moe_net_error().c_str());
        }
        g_moe.want_log = !params.moe_log.empty();
        fprintf(stderr, "nano-glm: routed experts via moe-server at %s\n", params.moe_addr.c_str());
        if (is_ds4) {
            moe_hello(ds4_moe_shape_of(M4.h), M4, params.moe_addr, params.moe_strict,
                      params.model_path, params.n_threads);
        } else {
            moe_hello(moe_shape_of(M.h), M, params.moe_addr, params.moe_strict,
                      params.model_path, params.n_threads);
        }
    }

    // Before the first eval: the trace hooks itself into the graph as it is built.
    if (!params.expert_log.empty()) {
        if (is_ds4) NANO_ABORT("--expert-log is not wired for deepseek4 yet");
        if (!expert_trace_built()) {
            NANO_ABORT("--expert-log needs a routing-trace build: .\\build.ps1 -Trace "
                       "(cmake -DNANO_EXPERT_TRACE=ON), then run build-trace\\bin\\nano-glm");
        }
        if (g_moe.active()) {
            // The router runs on the backend in that mode, so there is nothing
            // to observe here. moe-server would have to trace it, which needs
            // sequence positions the protocol does not carry.
            NANO_ABORT("--expert-log requires the local MoE path (drop --moe-addr)");
        }
        expert_trace_open(params.expert_log, moe_shape_of(M.h), M.desc, n_prompt);
        fprintf(stderr, "nano-glm: routing trace -> %s\n", params.expert_log.c_str());
    }

    const uint32_t kv_size = std::max((uint32_t) params.n_ctx, (uint32_t) (n_prompt + params.n_predict));

    nano_state S;
    ds4_state  S4;
    eval_ctx   Ectx;
    if (is_ds4) {
        ds4_init_state(S4, M4, kv_size, params.n_batch, params.n_threads);
        E4.eval = [&](const int32_t * t, int32_t n, int32_t past, bool all) {
            ds4_eval_chunk(M4, S4, t, n, past, all);
            return S4.logits.data();
        };
    } else {
        init_state(S, M, kv_size, params.n_threads);
        E4.eval = [&](const int32_t * t, int32_t n, int32_t past, bool all) {
            // glm-dsa always produces every position; `all` is the superset it
            // already gives, and a single-position caller reads row n-1.
            (void) all;
            eval_chunk(M, S, Ectx, t, n, past);
            return Ectx.logits.data();
        };
    }
    const auto t_load_end = std::chrono::steady_clock::now();

    fprintf(stderr, "nano-glm: %s\n", nano_build_line().c_str());
    fprintf(stderr, "nano-glm: %s | n_vocab=%u n_layer=%u n_embd=%u | n_prompt=%d n_predict=%d top_k=%d kv=%u threads=%d (%d physical / %d logical cores)\n",
            E4.desc.c_str(), E4.n_vocab, E4.n_layer, E4.n_embd,
            n_prompt, params.n_predict, params.top_k, kv_size, params.n_threads,
            physical_core_count(), (int) std::thread::hardware_concurrency());
    fprintf(stderr, "nano-glm: load+init %.1fs (mmap is lazy; first eval pages weights in)\n",
            std::chrono::duration<double>(t_load_end - t_load_start).count());

    std::vector<int32_t>       tokens = prompt;
    std::vector<lkld_position> positions;
    positions.reserve(n_prompt + params.n_predict);
    std::vector<int32_t> idx_buf;

    // prompt, chunked like collect (-b), all positions recorded
    const auto t_prompt_start = std::chrono::steady_clock::now();
    for (int32_t start = 0; start < n_prompt; ) {
        const int32_t end = std::min(n_prompt, start + params.n_batch);
        const float * lg = E4.eval(prompt.data() + start, end - start, start, /*all =*/ true);
        for (int32_t i = 0; i < end - start; i++) {
            positions.push_back(extract_topk_lse(lg + (size_t) i * E4.n_vocab,
                                                 E4.n_vocab, params.top_k, idx_buf));
        }
        start = end;
    }
    const auto t_prompt_end = std::chrono::steady_clock::now();

    // greedy generation: next token is the stored top-1 of the previous position
    const char * stop_reason = params.n_predict > 0 ? "length" : "none";
    int32_t next = positions.back().ids[0];
    for (int32_t step = 0; step < params.n_predict; step++) {
        tokens.push_back(next);
        fprintf(stdout, "%d ", next);
        fflush(stdout);

        const float * lg = E4.eval(&next, 1, n_prompt + step, /*all =*/ false);
        positions.push_back(extract_topk_lse(lg, E4.n_vocab, params.top_k, idx_buf));

        if (next == E4.eos_id) {
            stop_reason = "eos";
            break;
        }
        next = positions.back().ids[0];
    }
    const auto t_gen_end = std::chrono::steady_clock::now();
    fprintf(stdout, "\n");

    const int32_t n_total = (int32_t) tokens.size();
    const int32_t n_gen   = n_total - n_prompt;

    lkld_file out;
    out.n_vocab    = (int32_t) E4.n_vocab;
    out.top_k      = std::min(params.top_k, (int32_t) E4.n_vocab);
    out.model_desc = params.model_path + " | " + E4.desc;
    out.seqs.push_back({label, n_prompt, n_total, std::move(tokens), std::move(positions)});

    if (!lkld_write(params.output_path, out)) return 1;

    double tail_mean = 0.0, tail_max = 0.0;
    for (const lkld_position & p : out.seqs[0].positions) {
        double t = tail_mass(p);
        tail_mean += t;
        tail_max = std::max(tail_max, t);
    }
    tail_mean /= out.seqs[0].positions.size();

    const double prompt_s = std::chrono::duration<double>(t_prompt_end - t_prompt_start).count();
    const double gen_s    = std::chrono::duration<double>(t_gen_end - t_prompt_end).count();
    fprintf(stderr, "nano-glm: n_prompt=%d (%.1f tok/s), n_gen=%d (%.2f tok/s), stop=%s\n",
            n_prompt, n_prompt / prompt_s, n_gen, n_gen > 0 ? n_gen / gen_s : 0.0, stop_reason);
    fprintf(stderr, "nano-glm: top-%d tail mass: mean=%.3e max=%.3e\n", out.top_k, tail_mean, tail_max);
    fprintf(stderr, "nano-glm: wrote %s (%d positions)\n", params.output_path.c_str(),
            (int32_t) out.seqs[0].positions.size());

    if (!params.expert_log.empty()) {
        expert_trace_close();
        fprintf(stderr, "nano-glm: wrote %s (%" PRIu64 " positions x %u MoE layers)\n",
                params.expert_log.c_str(), expert_trace_n_pos(), M.h.n_layer - M.h.n_dense_lead);
    }

    if (g_moe.active()) {
        const moe_stats & st = g_moe.st;
        std::vector<uint32_t> rtt = g_moe.rtt_us;
        std::sort(rtt.begin(), rtt.end());
        const size_t n = rtt.size();
        if (n) {
            const size_t i90 = (size_t)(n * 0.9) < n ? (size_t)(n * 0.9) : n - 1;
            fprintf(stderr,
                    "nano-glm: MoE RPC: %" PRIu64 " calls, rtt p50 %u us p90 %u us max %u us\n",
                    st.n_calls, rtt[n / 2], rtt[i90], rtt[n - 1]);
            fprintf(stderr,
                    "nano-glm: MoE RPC: %.1fs total = %.1fs server + %.1fs network+queueing "
                    "(%.1f%%), %.1f MB out / %.1f MB in\n",
                    st.rtt_us / 1e6, st.srv_us / 1e6, (double)(st.rtt_us - st.srv_us) / 1e6,
                    st.rtt_us ? 100.0 * (double)(st.rtt_us - st.srv_us) / (double) st.rtt_us : 0.0,
                    st.bytes_out / 1e6, st.bytes_in / 1e6);
        }
        if (!params.moe_log.empty()) {
            FILE * f = fopen(params.moe_log.c_str(), "w");
            if (!f) NANO_ABORT("cannot write %s", params.moe_log.c_str());
            for (const moe_rpc_record & r : g_moe.log) {
                fprintf(f,
                        "{\"layer\":%u,\"n_tokens\":%u,\"bytes_out\":%u,\"bytes_in\":%u,"
                        "\"rtt_us\":%u,\"srv_parse_us\":%u,\"srv_route_us\":%u,"
                        "\"srv_compute_us\":%u,\"srv_serialize_us\":%u}\n",
                        r.layer, r.n_tokens, r.bytes_out, r.bytes_in, r.rtt_us,
                        r.srv_parse_us, r.srv_route_us, r.srv_compute_us, r.srv_ser_us);
            }
            fclose(f);
            fprintf(stderr, "nano-glm: wrote %s (%zu RPC records)\n", params.moe_log.c_str(), n);
        }
        moe_close(g_moe.sock);
    }

    return 0;
}
