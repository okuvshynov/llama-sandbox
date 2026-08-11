// dump — capture llama.cpp's named intermediate tensors for one forward pass.
//
// Why this exists: porting an architecture into nano-glm is otherwise
// verifiable only at the logits, and this project has already paid to learn
// that end-to-end logit KL saturates — 75 layers amplify any perturbation to
// the same ceiling, so a subtly wrong kernel and a correct one produce the
// same number (nano-glm/OPTIMIZATION.md). The same argument that made
// `moe-server --compare` necessary makes this necessary: compare at the
// tensor, not at the end.
//
// llama.cpp names every intermediate it builds (`cb(cur, "attn_norm", il)`)
// and exposes `ggml_backend_sched_eval_callback`, so the reference values are
// already there for the asking. This asks.
//
// Output is a flat, self-describing file — every tensor converted to F32 so a
// reader needs no quant knowledge, and the shapes carried alongside so a
// mismatch is a diagnosis rather than a puzzle. `dump_inspect.py` reads it.
//
//   dump -m model.gguf -p "text" --layer 0 -o ref-l0.ntd
//   dump -m model.gguf -p "text" --name attn_norm -o ref-norms.ntd
//
// A whole forward pass of every tensor is far too much to keep (one 4096x4x5
// tensor is 320 KB and there are hundreds), so a filter is not a convenience:
// without one this writes gigabytes. `--layer` is the filter to reach for,
// because porting proceeds one layer at a time.

#include "llama.h"
#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"

#include "cpu_topology.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// File format. Deliberately trivial: a magic, a count, then records. No
// index — a reader streams it, and the files are small by construction
// because the filters keep them so.
//
//   char   magic[8] = "NTDUMP1\0"
//   u32    n_records
//   record:
//     u32   name_len, then name bytes (no terminator)
//     i32   src_type      (ggml_type the tensor actually had)
//     i64   ne[4]
//     u64   n_elem        (product of ne, possibly truncated — see below)
//     f32   data[n_elem]
static const char NTD_MAGIC[8] = { 'N','T','D','U','M','P','1','\0' };

struct dump_params {
    std::string model_path;
    std::string prompt;
    std::string out_path = "dump.ntd";
    std::string name_filter;      // substring match on the tensor name
    int32_t     layer     = -2;   // -2 = any; -1 = the non-layer tensors
    int32_t     n_threads = (int32_t) physical_core_count();
    int32_t     n_ctx     = 4096;
    uint64_t    max_elem  = 4u * 1024 * 1024;  // per tensor, then truncate
};

struct dump_state {
    const dump_params * p = nullptr;
    FILE *   f = nullptr;
    uint32_t n_records = 0;
    std::vector<float>   f32;
    std::vector<uint8_t> raw;
};

// llama.cpp tags layer tensors as "name-il" (see llm_graph_context::cb).
// Matching on that suffix is how `--layer` selects one layer's worth.
static bool name_matches(const dump_params & p, const char * name) {
    if (!p.name_filter.empty() && !strstr(name, p.name_filter.c_str())) {
        return false;
    }
    if (p.layer == -2) {
        return true;
    }
    char suffix[32];
    snprintf(suffix, sizeof(suffix), "-%d", p.layer);
    const size_t n = strlen(name), s = strlen(suffix);
    if (p.layer >= 0) {
        return n > s && strcmp(name + n - s, suffix) == 0;
    }
    // -1: the tensors llama.cpp builds outside the layer loop, which it also
    // tags "-1". Same rule.
    return n > 2 && strcmp(name + n - 2, "-1") == 0;
}

static bool eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    dump_state & S = *(dump_state *) user_data;

    if (ask) {
        // Asked whether we want this one *before* it is computed. Saying no is
        // free; saying yes costs a device->host copy below.
        return name_matches(*S.p, t->name);
    }

    const int64_t n_elem_full = ggml_nelements(t);
    const uint64_t n_elem = std::min<uint64_t>((uint64_t) n_elem_full, S.p->max_elem);

    // The tensor may live on a non-CPU backend and may be quantised. Pull the
    // bytes, then convert to F32 through ggml's own type traits so the reader
    // never needs to know what MXFP4 is.
    S.raw.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, S.raw.data(), 0, ggml_nbytes(t));

    S.f32.resize((size_t) n_elem);
    if (t->type == GGML_TYPE_F32) {
        memcpy(S.f32.data(), S.raw.data(), (size_t) n_elem * sizeof(float));
    } else if (t->type == GGML_TYPE_I32) {
        const int32_t * src = (const int32_t *) S.raw.data();
        for (uint64_t i = 0; i < n_elem; i++) S.f32[i] = (float) src[i];
    } else {
        const ggml_type_traits * tr = ggml_get_type_traits(t->type);
        if (!tr || !tr->to_float) {
            fprintf(stderr, "dump: %s has type %s with no to_float; skipped\n",
                    t->name, ggml_type_name(t->type));
            return true;
        }
        // to_float works in whole blocks, so convert everything then truncate.
        std::vector<float> all((size_t) n_elem_full);
        tr->to_float(S.raw.data(), all.data(), n_elem_full);
        memcpy(S.f32.data(), all.data(), (size_t) n_elem * sizeof(float));
    }

    const uint32_t name_len = (uint32_t) strlen(t->name);
    const int32_t  src_type = (int32_t) t->type;
    fwrite(&name_len, sizeof(name_len), 1, S.f);
    fwrite(t->name, 1, name_len, S.f);
    fwrite(&src_type, sizeof(src_type), 1, S.f);
    fwrite(t->ne, sizeof(int64_t), 4, S.f);
    fwrite(&n_elem, sizeof(n_elem), 1, S.f);
    fwrite(S.f32.data(), sizeof(float), (size_t) n_elem, S.f);
    S.n_records++;

    // A one-line summary per tensor, so a run is useful even without the file:
    // shape and a few moments are usually enough to see that two ports agree
    // or to spot a NaN the moment it appears.
    double sum = 0.0, amax = 0.0;
    bool   bad = false;
    for (uint64_t i = 0; i < n_elem; i++) {
        const float v = S.f32[i];
        if (!(v == v) || v > 3.0e38f || v < -3.0e38f) bad = true;
        sum += v;
        amax = std::max(amax, (double) (v < 0 ? -v : v));
    }
    printf("%-34s %-7s [%5" PRId64 " %4" PRId64 " %4" PRId64 " %2" PRId64 "]  sum %+.6e  max|.| %.6e%s\n",
           t->name, ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3],
           sum, amax, bad ? "   <-- NaN/inf" : "");
    return true;
}

static bool parse_args(int argc, char ** argv, dump_params & p) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if      (a == "-m"      && i + 1 < argc) p.model_path  = argv[++i];
        else if (a == "-p"      && i + 1 < argc) p.prompt      = argv[++i];
        else if (a == "-o"      && i + 1 < argc) p.out_path    = argv[++i];
        else if (a == "--name"  && i + 1 < argc) p.name_filter = argv[++i];
        else if (a == "--layer" && i + 1 < argc) p.layer       = atoi(argv[++i]);
        else if (a == "-t"      && i + 1 < argc) p.n_threads   = atoi(argv[++i]);
        else if (a == "-c"      && i + 1 < argc) p.n_ctx       = atoi(argv[++i]);
        else if (a == "--max-elem" && i + 1 < argc) p.max_elem = strtoull(argv[++i], nullptr, 10);
        else { p.model_path.clear(); break; }
    }
    if (p.model_path.empty() || p.prompt.empty()) {
        fprintf(stderr,
            "Usage: dump -m <model.gguf> -p <prompt> [options]\n"
            "  -o <path>       output (default dump.ntd)\n"
            "  --layer <n>     only tensors llama.cpp tagged with this layer;\n"
            "                  -1 selects the ones built outside the layer loop\n"
            "  --name <sub>    only tensors whose name contains <sub>\n"
            "  --max-elem <n>  truncate each tensor to n elements (default 4M)\n"
            "  -c <int>        context size (default 4096)\n"
            "  -t <int>        threads (default: physical cores)\n"
            "\n"
            "Captures ONE forward pass over the prompt — no generation. Without a\n"
            "filter this writes every intermediate of every layer, which is\n"
            "gigabytes; --layer is the one to reach for.\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    dump_params params;
    if (!parse_args(argc, argv, params)) return 1;

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "dump: failed to load model '%s'\n", params.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(vocab, params.prompt, true, true);
    if (tokens.empty()) {
        fprintf(stderr, "dump: prompt tokenized to 0 tokens\n");
        llama_model_free(model);
        return 1;
    }

    dump_state S;
    S.p = &params;
    S.f = fopen(params.out_path.c_str(), "wb");
    if (!S.f) {
        fprintf(stderr, "dump: cannot write '%s'\n", params.out_path.c_str());
        llama_model_free(model);
        return 1;
    }
    uint32_t placeholder = 0;
    fwrite(NTD_MAGIC, 1, sizeof(NTD_MAGIC), S.f);
    fwrite(&placeholder, sizeof(placeholder), 1, S.f);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = std::max(params.n_ctx, (int32_t) tokens.size());
    ctx_params.n_batch         = (uint32_t) tokens.size();
    ctx_params.n_ubatch        = (uint32_t) tokens.size();
    ctx_params.n_threads       = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads;
    // The whole point. Note llama.cpp disables graph reuse when a callback is
    // set, so what runs is the freshly built graph, which is what we want to
    // read.
    ctx_params.cb_eval           = eval_cb;
    ctx_params.cb_eval_user_data = &S;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "dump: failed to create context\n");
        fclose(S.f); llama_model_free(model);
        return 1;
    }

    printf("dump: %zu tokens:", tokens.size());
    for (llama_token t : tokens) printf(" %d", t);
    printf("\n\n");

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], (llama_pos) i, { 0 }, i + 1 == tokens.size());
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);

    fseek(S.f, (long) sizeof(NTD_MAGIC), SEEK_SET);
    fwrite(&S.n_records, sizeof(S.n_records), 1, S.f);
    fclose(S.f);

    if (rc != 0) {
        fprintf(stderr, "dump: llama_decode failed (%d)\n", rc);
        llama_free(ctx); llama_model_free(model);
        return 1;
    }

    printf("\ndump: wrote %s (%u tensors)\n", params.out_path.c_str(), S.n_records);
    if (S.n_records == 0) {
        fprintf(stderr, "dump: nothing matched the filter — check --layer/--name "
                        "against the names printed by a run with no filter\n");
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
