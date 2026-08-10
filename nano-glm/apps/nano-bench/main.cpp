// nano-bench — throughput with a stated residency regime (PLAN.md step 11).
//
// Everything left in the plan is a claim about memory behaviour, and until now
// memory behaviour was the one thing that could not be measured: the same
// binary on the same model has produced 3.07 ms/layer warm against 52 ms cold,
// and 1.04-1.84 t/s across identical invocations, purely from what happened to
// be resident. An optimisation worth 10% is invisible against that.
//
// So this does not report "the" throughput. It reports throughput *in a named
// regime*, with every repetition printed, because the warmup curve is the
// signal and averaging it in is what produced the numbers above.
//
//   --hot    repeat the same single-token decode. One position's routing, so
//            the same 8-of-256 experts per layer every time: the smallest
//            working set the model can have, and the only regime where a TLB
//            effect is not buried under page-in.
//   --full   prefill and generate -n tokens, repeatedly. A 128-token window
//            already touches 171 of 256 experts per layer (ROUTING.md), so
//            this is close to the whole model and is the honest steady state.
//
// The gap between the two is the interesting part: if --hot is materially
// faster at equal bytes, keeping hot experts hot pays even with everything
// already in DRAM, which is a fact about step 3 that routing statistics alone
// cannot produce.
//
// Not a correctness tool. gate.py owns that, and the two must stay apart — a
// benchmark that also gated would invite trading determinism for speed.
//
// nano_graph.h first, for the winsock2-before-windows.h ordering.
#include "nano_graph.h"

#include "cpu_topology.h"
#include "nano_model.h"
#include "prompt_source.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

struct bench_params {
    std::string model_path;
    std::string input_bin;
    std::string tokens_str;
    bool        hot       = false;
    bool        full      = false;
    int32_t     n_predict = 256;   // --full only
    int32_t     n_reps    = 10;
    int32_t     n_ctx     = 4096;
    int32_t     n_batch   = 512;
    int32_t     n_threads = (int32_t) physical_core_count();
};

static bool parse_args(int argc, char ** argv, bench_params & p) {
    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        if      (!strcmp(a, "-m") && i + 1 < argc) p.model_path = argv[++i];
        else if (!strcmp(a, "-i") && i + 1 < argc) p.input_bin  = argv[++i];
        else if (!strcmp(a, "-T") && i + 1 < argc) p.tokens_str = argv[++i];
        else if (!strcmp(a, "-n") && i + 1 < argc) p.n_predict  = atoi(argv[++i]);
        else if (!strcmp(a, "-r") && i + 1 < argc) p.n_reps     = atoi(argv[++i]);
        else if (!strcmp(a, "-c") && i + 1 < argc) p.n_ctx      = atoi(argv[++i]);
        else if (!strcmp(a, "-b") && i + 1 < argc) p.n_batch    = atoi(argv[++i]);
        else if (!strcmp(a, "-t") && i + 1 < argc) p.n_threads  = atoi(argv[++i]);
        else if (!strcmp(a, "--hot"))  p.hot  = true;
        else if (!strcmp(a, "--full")) p.full = true;
        else {
            fprintf(stderr, "nano-bench: unknown argument '%s'\n", a);
            p.model_path.clear();
            break;
        }
    }
    if (p.model_path.empty() || (p.input_bin.empty() == p.tokens_str.empty())
            || (p.hot == p.full)) {
        fprintf(stderr,
            "Usage: nano-bench -m <first-shard.gguf> (-i <lkldtopk.bin> | -T <ids>)\n"
            "                  (--hot | --full) [options]\n"
            "  --hot       repeat one single-token decode: smallest working set\n"
            "  --full      prefill + generate -n tokens, repeatedly: whole model\n"
            "  -n <int>    tokens per repetition for --full (default: 256)\n"
            "  -r <int>    repetitions (default: 10)\n"
            "  -c/-b/-t    context, prompt chunk, threads — as nano-glm\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// byte budget, from the model that is actually loaded
//
// Every throughput claim in PLAN.md rests on a bytes-per-token figure, and
// those were derived by hand from tensor offsets. Deriving them here instead
// means the GB/s below is right for whatever GGUF is open, and that a quant
// change updates the budget rather than silently invalidating it.

struct byte_budget {
    double dense = 0;     // read in full every token: attention, norms, shared expert, lm_head
    double shexp = 0;     // part of dense, broken out because it is the MoE half of it
    double routed = 0;    // all routed experts; only n_used/n_expert of it per token
    double embd = 0;      // token_embd: one row per token, so ~nothing — excluded
    double per_token() const { return dense + routed_per_token(); }
    double routed_per_token() const { return routed_frac * routed; }
    double routed_frac = 0;
    bool   tied = false;  // lm_head shares the embedding table
};

static byte_budget model_bytes(const nano_model & M) {
    byte_budget b;
    b.routed_frac = (double) M.h.n_expert_used / (double) M.h.n_expert;

    // With tied embeddings, load_model points M.output at token_embd, so the
    // lm_head matmul reads the whole table every token and it is dense, not a
    // one-row lookup. GLM-5.2 ships a separate output.weight, but getting this
    // backwards on a tied model would understate the budget by the size of the
    // vocabulary — 1.01 GB here, 2.6% — in the silent direction.
    const bool tied = (M.output == M.tok_embd);

    for (const auto & kv : M.tensors) {
        const double n = (double) ggml_nbytes(kv.second);
        if (kv.first.find("_exps") != std::string::npos) {
            b.routed += n;
        } else if (kv.first == "token_embd.weight" && !tied) {
            b.embd += n;                      // get_rows only: one row per token
        } else {
            b.dense += n;
            if (kv.first.find("_shexp") != std::string::npos) b.shexp += n;
        }
    }
    b.tied = tied;
    return b;
}

// ---------------------------------------------------------------------------
// reporting

static void report(const char * what, std::vector<double> & ms, double bytes_per_token) {
    if (ms.empty()) return;
    // Steady state is the back half. The first repetitions are the page-in
    // curve, and folding them into one mean is precisely the habit this tool
    // exists to break — so they are printed above and excluded here, rather
    // than quietly averaged.
    const size_t half = ms.size() / 2;
    std::vector<double> tail(ms.begin() + half, ms.end());
    std::sort(tail.begin(), tail.end());
    const double med = tail[tail.size() / 2];
    const double p25 = tail[tail.size() / 4];
    const double p75 = tail[(tail.size() * 3) / 4];

    fprintf(stdout, "  %s: median %.1f ms/token, %.3f tok/s, %.1f GB/s\n",
            what, med, 1000.0 / med, bytes_per_token / (med / 1000.0) / 1e9);
    // Both spreads on purpose. The full range answers "how bad can one
    // repetition be", which is what a single-shot timing is exposed to; the
    // interquartile range answers "what can I resolve", and a lone OS hiccup
    // should not be allowed to say a 10% optimisation is unmeasurable.
    fprintf(stdout, "  last %zu of %zu reps: p25-p75 %.1f-%.1f ms (%.1f%%), "
                    "full range %.1f-%.1f ms (%.1f%%)\n",
            tail.size(), ms.size(), p25, p75, 100.0 * (p75 - p25) / med,
            tail.front(), tail.back(), 100.0 * (tail.back() - tail.front()) / med);
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--version")) {
            fputs(nano_build_info().c_str(), stdout);
            fputs(nano_run_info((int) physical_core_count()).c_str(), stdout);
            return 0;
        }
    }

    bench_params p;
    if (!parse_args(argc, argv, p)) return 1;

    std::string label;
    std::vector<int32_t> prompt = load_prompt_tokens(p.input_bin, p.tokens_str, label, "nano-bench");
    const int32_t n_prompt = (int32_t) prompt.size();

    nano_model M;
    const auto t0 = clk::now();
    load_model(M, p.model_path);
    const nano_hparams & h = M.h;

    const uint32_t kv_size = std::max((uint32_t) p.n_ctx,
                                      (uint32_t) (n_prompt + p.n_predict + 1));
    nano_state S;
    init_state(S, M, kv_size, p.n_threads);

    const byte_budget B = model_bytes(M);

    fprintf(stdout, "nano-bench: %s\n", nano_build_line().c_str());
    fprintf(stdout, "  model    %s | %u shards, %.2f GB mapped\n",
            M.desc.c_str(), M.n_shards, M.bytes_mapped / 1e9);
    fprintf(stdout, "  regime   %s | prompt %d tokens (%s) | %d reps | %d threads\n",
            p.hot ? "hot (one position, same experts every rep)"
                  : "full (prefill + generate, whole model)",
            n_prompt, label.c_str(), p.n_reps, p.n_threads);
    fprintf(stdout, "  bytes/tok %.2f GB = dense %.2f (of which shared expert %.2f) "
                    "+ routed %.2f (%u/%u of %.2f GB)\n",
            B.per_token() / 1e9, B.dense / 1e9, B.shexp / 1e9,
            B.routed_per_token() / 1e9, h.n_expert_used, h.n_expert, B.routed / 1e9);
    if (B.tied) {
        fprintf(stdout, "  (lm_head is tied to the embedding table, so it counts as dense; "
                        "KV cache traffic is excluded)\n");
    } else {
        fprintf(stdout, "  (KV cache traffic and the %.2f GB embedding table are excluded: "
                        "get_rows reads one row per token)\n", B.embd / 1e9);
    }
    fprintf(stdout, "  load+init %.1fs\n\n", secs(t0, clk::now()));
    // Redirected stdout is fully buffered, and the first repetition can take
    // minutes; without this the byte budget appears only once the run ends,
    // which is the least useful moment for it.
    fflush(stdout);

    eval_ctx E;
    auto argmax = [&](const float * logits) {
        int32_t best = 0;
        for (uint32_t i = 1; i < h.n_vocab; i++) if (logits[i] > logits[best]) best = (int32_t) i;
        return best;
    };

    if (p.hot) {
        // Prefill once, then re-decode the same position over and over. The
        // KV slot is rewritten with identical values each time, so routing —
        // and therefore the working set — is bit-identical across reps.
        int32_t last = 0;
        for (int32_t start = 0; start < n_prompt; ) {
            const int32_t end = std::min(n_prompt, start + p.n_batch);
            last = end - start;
            eval_chunk(M, S, E, prompt.data() + start, last, start);
            start = end;
        }
        const int32_t tok = argmax(E.logits.data() + (size_t) (last - 1) * h.n_vocab);

        std::vector<double> ms;
        for (int32_t r = 0; r < p.n_reps; r++) {
            const auto a = clk::now();
            eval_chunk(M, S, E, &tok, 1, n_prompt);
            const double dt = secs(a, clk::now()) * 1000.0;
            ms.push_back(dt);
            fprintf(stdout, "  rep %3d  %8.1f ms  %6.3f tok/s  %6.1f GB/s\n",
                    r + 1, dt, 1000.0 / dt, B.per_token() / (dt / 1000.0) / 1e9);
            fflush(stdout);
        }
        fprintf(stdout, "\n");
        report("hot decode", ms, B.per_token());
    } else {
        std::vector<double> ms;
        for (int32_t r = 0; r < p.n_reps; r++) {
            // Restart at position 0: the causal mask ignores everything past
            // the current position, so stale cells below it are unreachable.
            const auto a = clk::now();
            int32_t last = 0;
            for (int32_t start = 0; start < n_prompt; ) {
                const int32_t end = std::min(n_prompt, start + p.n_batch);
                last = end - start;
                eval_chunk(M, S, E, prompt.data() + start, last, start);
                start = end;
            }
            const auto b = clk::now();

            int32_t next = argmax(E.logits.data() + (size_t) (last - 1) * h.n_vocab);
            for (int32_t i = 0; i < p.n_predict; i++) {
                eval_chunk(M, S, E, &next, 1, n_prompt + i);
                next = argmax(E.logits.data());
            }
            const auto c = clk::now();

            const double gen_ms = secs(b, c) * 1000.0;
            ms.push_back(gen_ms / p.n_predict);
            fprintf(stdout, "  rep %3d  prefill %6.1fs (%5.1f tok/s)  generate %6.1fs "
                            "(%5.3f tok/s, %5.1f GB/s)\n",
                    r + 1, secs(a, b), n_prompt / secs(a, b), secs(b, c),
                    p.n_predict / secs(b, c),
                    B.per_token() / (gen_ms / p.n_predict / 1000.0) / 1e9);
            fflush(stdout);
        }
        fprintf(stdout, "\n");
        report("generate, per token", ms, B.per_token());
    }
    return 0;
}
