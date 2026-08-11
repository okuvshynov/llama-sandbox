// nano-chat — single-turn completion with a tokenizer and a chat template.
//
// The other side of the split PLAN.md step 7 draws. `nano-glm` takes token ids
// and emits logits, and its interface is frozen because the bit-exactness
// contract is defined over a fixed token sequence. Everything that can *change*
// that sequence — tokenizer, chat template, sampling — lives here instead, so
// adding any of it cannot invalidate a stored reference.
//
// Greedy for now, so a run is still reproducible; a sampler is the first thing
// that would make it not, and it belongs here when it arrives.
//
// models/glm_dsa/graph.h first: it reaches moe_proto.h, and winsock2.h must precede the
// windows.h that models/glm_dsa/model.h pulls in.
#include "models/glm_dsa/graph.h"

#include "models/glm_dsa/chat.h"
#include "cpu_topology.h"
#include "models/glm_dsa/model.h"
#include "vocab.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#   include <shellapi.h>   // CommandLineToArgvW; shell32 is already a default lib
#   include <io.h>
#endif

// Windows hands main() an argv that the CRT converted from the wide command
// line through the *ANSI* code page. On this machine that is cp1252, so "ç"
// arrives as the single byte 0xE7 instead of UTF-8 C3 A7, and anything outside
// cp1252 — every CJK character — arrives as "?". A byte-level BPE then
// tokenizes the mangled bytes perfectly happily and returns ids that are
// simply wrong, with nothing anywhere to say so.
//
// Found by tokenizer_check.py: every non-ASCII case disagreed with llama.cpp
// and every ASCII case matched, which is the signature of an encoding problem
// rather than a tokenizer one. Read the real command line instead.
static std::vector<std::string> utf8_args(int argc, char ** argv) {
#if defined(_WIN32)
    int wargc = 0;
    LPWSTR * wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv) NANO_ABORT("CommandLineToArgvW failed (err %lu)", GetLastError());
    std::vector<std::string> out;
    out.reserve(wargc);
    for (int i = 0; i < wargc; i++) {
        const int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(n > 1 ? n - 1 : 0, '\0');
        if (n > 1) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], n, nullptr, nullptr);
        out.push_back(s);
    }
    LocalFree(wargv);
    return out;
#else
    return std::vector<std::string>(argv, argv + argc);
#endif
}

struct chat_params {
    std::string model_path;
    std::string prompt;
    std::string system;
    int32_t     n_predict = 512;
    int32_t     n_ctx     = 8192;
    int32_t     n_batch   = 512;
    int32_t     n_threads = (int32_t) physical_core_count();
    std::string moe_addr;
    bool        moe_strict = false;
    bool        think      = true;   // template default
    bool        raw        = false;  // no chat template: plain completion
    bool        dry_run    = false;  // tokenize and exit, no weights touched
};

static bool parse_args(const std::vector<std::string> & args, chat_params & p) {
    const int argc = (int) args.size();
    std::string prompt_file;
    for (int i = 1; i < argc; i++) {
        const std::string & a = args[i];
        if      (a == "-m"  && i + 1 < argc) p.model_path = args[++i];
        else if (a == "-p"  && i + 1 < argc) p.prompt     = args[++i];
        else if (a == "-f"  && i + 1 < argc) prompt_file  = args[++i];
        else if (a == "-s"  && i + 1 < argc) p.system     = args[++i];
        else if (a == "-n"  && i + 1 < argc) p.n_predict  = atoi(args[++i].c_str());
        else if (a == "-c"  && i + 1 < argc) p.n_ctx      = atoi(args[++i].c_str());
        else if (a == "-b"  && i + 1 < argc) p.n_batch    = atoi(args[++i].c_str());
        else if (a == "-t"  && i + 1 < argc) p.n_threads  = atoi(args[++i].c_str());
        else if (a == "--moe-addr" && i + 1 < argc) p.moe_addr = args[++i];
        else if (a == "--strict")   p.moe_strict = true;
        else if (a == "--no-think") p.think      = false;
        else if (a == "--raw")      p.raw        = true;
        else if (a == "--dry-run")  p.dry_run    = true;
        else {
            fprintf(stderr, "nano-chat: unknown argument '%s'\n", a.c_str());
            p.model_path.clear();
            break;
        }
    }
    if (!prompt_file.empty()) {
        // Bytes straight off disk: no code page anywhere in the path, and no
        // command-line length limit either.
        FILE * f = fopen(prompt_file.c_str(), "rb");
        if (!f) NANO_ABORT("cannot read prompt file '%s'", prompt_file.c_str());
        char buf[4096];
        size_t n;
        p.prompt.clear();
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) p.prompt.append(buf, n);
        fclose(f);
    }
    if (p.model_path.empty() || p.prompt.empty()) {
        fprintf(stderr,
            "Usage: nano-chat -m <first-shard.gguf> (-p <prompt> | -f <file>) [options]\n"
            "  -f <path>   read the prompt from a file (UTF-8 bytes, no length limit)\n"
            "  -s <text>   system message (optional)\n"
            "  -n <int>    tokens to generate, greedy (default: 512)\n"
            "  -c <int>    context size, auto-raised to fit (default: 8192)\n"
            "  -b <int>    prompt chunk size (default: 512)\n"
            "  -t <int>    threads (default: physical cores, ignoring SMT siblings)\n"
            "  --no-think  ask the model to skip reasoning (<think></think>)\n"
            "  --raw       no chat template: continue the prompt verbatim\n"
            "  --dry-run   print the prompt tokenization and exit; no weights are\n"
            "              read, so this costs a second and is how you check what\n"
            "              the template actually built\n"
            "  --moe-addr <host:port> / --strict   as nano-glm\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
#if defined(_WIN32)
    // The model emits UTF-8; without this the console renders it as cp1252
    // mojibake. Harmless when stdout is redirected, which is why it is easy to
    // miss until someone actually watches a generation.
    SetConsoleOutputCP(CP_UTF8);
#endif

    chat_params params;
    if (!parse_args(utf8_args(argc, argv), params)) return 1;

    // The vocab lives in shard 1 (9.4 MB of metadata), so tokenizing costs a
    // second and does not touch the 583 GiB of weights behind it.
    nano_vocab V;
    load_vocab(V, params.model_path);

    const std::vector<int32_t> prompt = params.raw
        ? tokenize(V, params.prompt)
        : glm_chat_prompt(V, params.prompt, params.system, params.think);

    if (params.dry_run) {
        std::string flat;
        for (int32_t t : prompt) flat += detokenize(V, t);
        fprintf(stderr, "nano-chat: %zu tokens\n", prompt.size());
        fprintf(stderr, "nano-chat: prompt reads as:\n%s\n", flat.c_str());
        for (size_t i = 0; i < prompt.size(); i++) {
            fprintf(stdout, "%s%d", i ? "," : "", prompt[i]);
        }
        fprintf(stdout, "\n");
        return 0;
    }

    nano_model M;
    const auto t_load0 = std::chrono::steady_clock::now();
    load_model(M, params.model_path);
    const nano_hparams & h = M.h;

    if (!params.moe_addr.empty()) {
        const size_t colon = params.moe_addr.rfind(':');
        if (colon == std::string::npos) NANO_ABORT("--moe-addr must be host:port");
        if (!moe_net_init()) NANO_ABORT("socket init failed");
        g_moe.sock = moe_connect(params.moe_addr.substr(0, colon),
                                 (uint16_t) atoi(params.moe_addr.c_str() + colon + 1));
        if (!g_moe.active()) {
            NANO_ABORT("cannot reach moe-server at %s (%s)", params.moe_addr.c_str(),
                       moe_net_error().c_str());
        }
        fprintf(stderr, "nano-chat: routed experts via moe-server at %s\n", params.moe_addr.c_str());
        moe_hello(moe_shape_of(M.h), M, params.moe_addr, params.moe_strict, params.model_path, params.n_threads);
    }

    const int32_t n_prompt = (int32_t) prompt.size();
    const uint32_t kv_size = std::max((uint32_t) params.n_ctx,
                                      (uint32_t) (n_prompt + params.n_predict));
    nano_state S;
    init_state(S, M, kv_size, params.n_threads);

    fprintf(stderr, "nano-chat: %s | %s | n_prompt=%d n_predict=%d kv=%u threads=%d\n",
            M.desc.c_str(), nano_build_line().c_str(), n_prompt, params.n_predict,
            kv_size, params.n_threads);
    fprintf(stderr, "nano-chat: load+init %.1fs (mmap is lazy; the first eval pages weights in)\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count());

    eval_ctx E;

    // prefill, chunked exactly like nano-glm so the two agree token for token
    const auto t_prompt0 = std::chrono::steady_clock::now();
    int32_t last_chunk = 0;
    for (int32_t start = 0; start < n_prompt; ) {
        const int32_t end = std::min(n_prompt, start + params.n_batch);
        last_chunk = end - start;
        eval_chunk(M, S, E, prompt.data() + start, last_chunk, start);
        start = end;
    }
    const auto t_prompt1 = std::chrono::steady_clock::now();

    // greedy: argmax over the last position's logits
    auto argmax = [&](const float * logits) {
        int32_t best = 0;
        for (uint32_t i = 1; i < h.n_vocab; i++) if (logits[i] > logits[best]) best = (int32_t) i;
        return best;
    };

    // E.logits holds the *last chunk*, all of its positions; the one that
    // predicts the first generated token is its final row.
    int32_t next = argmax(E.logits.data() + (size_t) (last_chunk - 1) * h.n_vocab);
    const char * stop = "length";
    int32_t n_gen = 0;

    for (int32_t step = 0; step < params.n_predict; step++) {
        if (next == V.eos_id || next == V.eot_id) { stop = "eos"; break; }

        // Stream: a token is bytes, not necessarily a whole character, so this
        // writes raw and lets the terminal reassemble multi-byte sequences.
        const std::string piece = detokenize(V, next);
        fwrite(piece.data(), 1, piece.size(), stdout);
        fflush(stdout);
        n_gen++;

        eval_chunk(M, S, E, &next, 1, n_prompt + step);
        next = argmax(E.logits.data());
    }
    const auto t_gen1 = std::chrono::steady_clock::now();
    fprintf(stdout, "\n");

    const double prompt_s = std::chrono::duration<double>(t_prompt1 - t_prompt0).count();
    const double gen_s    = std::chrono::duration<double>(t_gen1 - t_prompt1).count();
    fprintf(stderr, "nano-chat: prompt %d (%.1f tok/s), generated %d (%.2f tok/s), stop=%s\n",
            n_prompt, n_prompt / prompt_s, n_gen, n_gen > 0 ? n_gen / gen_s : 0.0, stop);

    if (g_moe.active()) moe_close(g_moe.sock);
    return 0;
}
