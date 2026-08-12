// ds4-port — run nano-glm's DeepSeek-V4-Flash trunk as far as it is ported and
// dump every named tensor, in the same format `logit-kld/src/dump.cpp` writes
// for llama.cpp.
//
// The porting harness, and temporary by design: it exists so each tensor can be
// checked against the reference the moment it is written, and it goes away once
// the graph is complete and `gate.py llamacpp` takes over. Compare with:
//
//   dump.exe -m ds4.gguf -p "..." --layer 0 -o ref.ntd        (llama.cpp)
//   ds4-port -m ds4.gguf -T <ids>  -o mine.ntd                (this)
//   python dump_inspect.py ref.ntd mine.ntd
//
// Token ids rather than text: llama.cpp's `dump` prints the ids it tokenized,
// and passing those in removes the tokenizer from the comparison entirely. One
// variable at a time.
//
// moe_proto.h first: winsock2.h must precede the windows.h gguf_store.h pulls
// in, even though nothing here speaks the protocol.

#include "moe_proto.h"

#include "build_info.h"
#include "cpu_topology.h"
#include "models/deepseek4/graph.h"
#include "models/deepseek4/model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char NTD_MAGIC[8] = { 'N','T','D','U','M','P','1','\0' };

struct port_params {
    std::string model_path;
    std::string tokens_str;
    std::string out_path  = "ds4-port.ntd";
    int32_t     n_threads = physical_core_count();
    int32_t     n_layers  = 0;   // 0 = every layer the graph can build
    ds4_stage   stage     = DS4_STAGE_LAYER;
};

static bool parse_stage(const char * s, ds4_stage & out) {
    if (!strcmp(s, "hc_pre"))   { out = DS4_STAGE_HC_PRE;   return true; }
    if (!strcmp(s, "compress")) { out = DS4_STAGE_COMPRESS; return true; }
    if (!strcmp(s, "attn"))     { out = DS4_STAGE_ATTN;     return true; }
    if (!strcmp(s, "layer"))    { out = DS4_STAGE_LAYER;    return true; }
    return false;
}

static std::vector<int32_t> parse_tokens(const std::string & s) {
    std::vector<int32_t> out;
    const char * p = s.c_str();
    while (*p) {
        char * end = nullptr;
        const long v = strtol(p, &end, 10);
        if (end == p) NANO_ABORT("bad token list near '%s'", p);
        out.push_back((int32_t) v);
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    return out;
}

// One record, matching dump.cpp's format exactly: name, source type, ne[4],
// element count, then f32 data.
static void write_record(FILE * f, ggml_tensor * t, std::vector<float> & scratch) {
    const uint64_t n_elem = (uint64_t) ggml_nelements(t);
    scratch.resize((size_t) n_elem);

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, scratch.data(), 0, n_elem * sizeof(float));
    } else if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> tmp((size_t) n_elem);
        ggml_backend_tensor_get(t, tmp.data(), 0, n_elem * sizeof(int32_t));
        for (uint64_t i = 0; i < n_elem; i++) scratch[i] = (float) tmp[i];
    } else {
        fprintf(stderr, "ds4-port: %s has type %s, not dumped\n", t->name, ggml_type_name(t->type));
        return;
    }

    // llama.cpp tags names "<base>-<il>"; ours already do, so they match.
    const uint32_t name_len = (uint32_t) strlen(t->name);
    const int32_t  src_type = (int32_t) t->type;
    fwrite(&name_len, sizeof(name_len), 1, f);
    fwrite(t->name, 1, name_len, f);
    fwrite(&src_type, sizeof(src_type), 1, f);
    fwrite(t->ne, sizeof(int64_t), 4, f);
    fwrite(&n_elem, sizeof(n_elem), 1, f);
    fwrite(scratch.data(), sizeof(float), (size_t) n_elem, f);

    double sum = 0.0, amax = 0.0;
    bool bad = false;
    for (uint64_t i = 0; i < n_elem; i++) {
        const float v = scratch[i];
        if (!(v == v) || v > 3.0e38f || v < -3.0e38f) bad = true;
        sum += v;
        const double a = v < 0 ? -v : v;
        if (a > amax) amax = a;
    }
    printf("%-34s %-7s [%5" PRId64 " %4" PRId64 " %4" PRId64 " %2" PRId64 "]  sum %+.6e  max|.| %.6e%s\n",
           t->name, ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3],
           sum, amax, bad ? "   <-- NaN/inf" : "");
}

int main(int argc, char ** argv) {
    port_params p;
    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        if      (!strcmp(a, "-m") && i + 1 < argc) p.model_path = argv[++i];
        else if (!strcmp(a, "-T") && i + 1 < argc) p.tokens_str = argv[++i];
        else if (!strcmp(a, "-o") && i + 1 < argc) p.out_path   = argv[++i];
        else if (!strcmp(a, "-t") && i + 1 < argc) p.n_threads  = atoi(argv[++i]);
        else if (!strcmp(a, "-L") && i + 1 < argc) p.n_layers   = atoi(argv[++i]);
        else if (!strcmp(a, "--stage") && i + 1 < argc) {
            if (!parse_stage(argv[++i], p.stage)) { p.model_path.clear(); break; }
        }
        else { p.model_path.clear(); break; }
    }
    if (p.model_path.empty() || p.tokens_str.empty()) {
        fprintf(stderr,
            "Usage: ds4-port -m <first-shard.gguf> -T <id,id,...> [-o out.ntd] [-t threads] [-L layers]\n"
            "  Runs the deepseek4 trunk as far as it is ported and dumps every\n"
            "  named tensor, for comparison against logit-kld's `dump` of\n"
            "  llama.cpp. Token ids, not text: that keeps the tokenizer out of\n"
            "  the comparison.\n"
            "  -L defaults to every layer the graph can build; pass fewer to\n"
            "     bisect, since a layer's input is the previous layer's output.\n"
            "  --stage hc_pre|compress|attn|layer  how far into the LAST layer\n"
            "     to build (default layer). 'compress' is how a ratio-4 layer is\n"
            "     reached while only its compressors are ported.\n");
        return 1;
    }

    const std::vector<int32_t> tokens = parse_tokens(p.tokens_str);
    const int32_t n_tokens = (int32_t) tokens.size();

    ds4_model M;
    ds4_load_model(M, p.model_path);

    const uint32_t n_ported = ds4_ported_layers(M.h);
    if (p.n_layers <= 0) {
        p.n_layers = (int32_t) n_ported;
    } else if ((uint32_t) p.n_layers > M.h.n_layer) {
        NANO_ABORT("-L %d exceeds the model's %u layers", p.n_layers, M.h.n_layer);
    }
    // Beyond the range check the graph is the authority on what it can build:
    // it knows which stages a partially-ported layer supports, and duplicating
    // that policy here is how the two drift apart.

    fprintf(stderr, "ds4-port: %s | %s\n", M.desc.c_str(), nano_build_line().c_str());
    fprintf(stderr, "ds4-port: %d tokens, %d threads, %d layers to stage '%s' "
                    "(%u fully ported)\n",
            n_tokens, p.n_threads, p.n_layers, ds4_stage_name(p.stage), n_ported);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) NANO_ABORT("no CPU backend");
    ggml_backend_cpu_set_n_threads(backend, p.n_threads);

    std::vector<uint8_t> meta(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params ip = { meta.size(), meta.data(), /*no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) NANO_ABORT("no graph context");

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);

    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    // One prefill from position 0, so the mask is plainly causal. The model's
    // 128-wide sliding window does not bite at these lengths; when it does,
    // this harness needs it and the assert below is the reminder.
    if (n_tokens > (int32_t) M.h.sliding_window) {
        NANO_ABORT("%d tokens exceeds the %u sliding window; the mask here is causal only",
                   n_tokens, M.h.sliding_window);
    }
    // F16: ggml_flash_attn_ext requires it, where the explicit path wants F32.
    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_tokens, n_tokens);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);

    // ---- compressor state, for the one case this harness covers ----
    //
    // A ratio-4 layer folds every 4 tokens into one compressed key, and each
    // block also reads the *previous* block's tokens. In a real eval the
    // previous ones come out of the carried state; here the cache is empty and
    // the whole prompt arrives at once, so the state is analytic: zeros, and
    // block 0's history is the pad row.
    //
    // This is exactly the restriction the harness already lives under (one
    // prefill from position 0, inside the sliding window). It is also the
    // reason `n_blocks` and the index arrays are built here rather than in the
    // graph: a real KV cache computes the same tensors from real state, and
    // the graph should not have to change when it does.
    const int32_t ratio    = 4;
    const int32_t n_base   = 2 * ratio;              // rows llama.cpp carries
    const int32_t n_blocks = n_tokens / ratio;       // complete blocks only
    const int32_t n_read   = ratio * n_blocks;

    // Row i of the concatenated [base | current] tensor, plus one pad row that
    // `ds4_append_zero_row` puts at the end. Block b reads its own `ratio`
    // tokens and the previous block's; block 0 has no previous, so it reads the
    // pad row, which is zero for values and -inf for scores.
    std::vector<int32_t> read_idxs((size_t) 2 * n_read);
    for (int32_t b = 0; b < n_blocks; b++) {
        for (int32_t j = 0; j < ratio; j++) {
            const int32_t prev = b == 0 ? n_base + n_tokens          // the pad row
                                        : n_base + (b - 1) * ratio + j;
            read_idxs[(size_t) b * ratio + j]          = prev;
            read_idxs[(size_t) n_read + b * ratio + j] = n_base + b * ratio + j;
        }
    }

    // The rope position of a compressed key is the position of the **first
    // token of its block**, not the block index: llama.cpp pushes
    // `source_start = pos + 1 - ratio` for the token that closes the block
    // (llama-kv-cache-dsv4.cpp:527). Both are 0 for block 0, so a prompt short
    // enough for a single block cannot tell them apart — which is why this is
    // checked at 12 tokens and not 5.
    std::vector<int32_t> comp_pos((size_t) n_blocks);
    for (int32_t b = 0; b < n_blocks; b++) comp_pos[b] = b * ratio;

    std::vector<int32_t> ape_pos((size_t) n_tokens);
    for (int32_t i = 0; i < n_tokens; i++) ape_pos[i] = i % ratio;

    auto new_input = [&](ggml_type type, int64_t ne0, int64_t ne1, const char * name) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };

    ds4_layer_inputs lin;
    if (n_blocks > 0) {
        lin.csa.base_kv    = new_input(GGML_TYPE_F32, 2 * M.h.d_key, n_base, "csa_base_kv");
        lin.csa.base_score = new_input(GGML_TYPE_F32, 2 * M.h.d_key, n_base, "csa_base_score");
        lin.csa.read_idxs  = new_input(GGML_TYPE_I32, 2 * n_read, 1, "csa_read_idxs");
        lin.csa.comp_pos   = new_input(GGML_TYPE_I32, n_blocks, 1, "csa_comp_pos");

        lin.lid.base_kv    = new_input(GGML_TYPE_F32, 2 * M.h.idx_key_len, n_base, "lid_base_kv");
        lin.lid.base_score = new_input(GGML_TYPE_F32, 2 * M.h.idx_key_len, n_base, "lid_base_score");
        lin.lid.read_idxs  = new_input(GGML_TYPE_I32, 2 * n_read, 1, "lid_read_idxs");
        lin.lid.comp_pos   = new_input(GGML_TYPE_I32, n_blocks, 1, "lid_comp_pos");

        lin.ape_pos = new_input(GGML_TYPE_I32, n_tokens, 1, "ape_pos");
        lin.rot     = new_input(GGML_TYPE_F32, M.h.idx_key_len, M.h.idx_key_len, "lid_k_rot");
    } else if (p.stage == DS4_STAGE_COMPRESS) {
        NANO_ABORT("%d tokens make no complete %d-token block; a compressor "
                   "layer needs at least %d", n_tokens, ratio, ratio);
    }

    ggml_tensor * out = ds4_build_graph(ctx, gf, M, inp_tokens, inp_pos, kq_mask, lin,
                                        n_tokens, p.stage, (uint32_t) p.n_layers);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) NANO_ABORT("graph alloc failed");

    ggml_backend_tensor_set(inp_tokens, tokens.data(), 0, n_tokens * sizeof(int32_t));

    std::vector<int32_t> pos(n_tokens);
    for (int32_t i = 0; i < n_tokens; i++) pos[i] = i;
    ggml_backend_tensor_set(inp_pos, pos.data(), 0, n_tokens * sizeof(int32_t));

    // mask[j, i] is the bias added to the score of key j for query i.
    std::vector<ggml_fp16_t> mask((size_t) n_tokens * n_tokens);
    for (int32_t i = 0; i < n_tokens; i++) {
        for (int32_t j = 0; j < n_tokens; j++) {
            mask[(size_t) i * n_tokens + j] = ggml_fp32_to_fp16(j <= i ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    if (n_blocks > 0) {
        const std::vector<float> zeros_csa((size_t) 2 * M.h.d_key * n_base, 0.0f);
        const std::vector<float> zeros_lid((size_t) 2 * M.h.idx_key_len * n_base, 0.0f);
        auto set = [](ggml_tensor * t, const void * d, size_t n) {
            ggml_backend_tensor_set(t, d, 0, n);
        };
        set(lin.csa.base_kv,    zeros_csa.data(), zeros_csa.size() * sizeof(float));
        set(lin.csa.base_score, zeros_csa.data(), zeros_csa.size() * sizeof(float));
        set(lin.lid.base_kv,    zeros_lid.data(), zeros_lid.size() * sizeof(float));
        set(lin.lid.base_score, zeros_lid.data(), zeros_lid.size() * sizeof(float));

        set(lin.csa.read_idxs, read_idxs.data(), read_idxs.size() * sizeof(int32_t));
        set(lin.lid.read_idxs, read_idxs.data(), read_idxs.size() * sizeof(int32_t));
        set(lin.csa.comp_pos,  comp_pos.data(),  comp_pos.size()  * sizeof(int32_t));
        set(lin.lid.comp_pos,  comp_pos.data(),  comp_pos.size()  * sizeof(int32_t));
        set(lin.ape_pos,       ape_pos.data(),   ape_pos.size()   * sizeof(int32_t));

        // Order 128, not d_key: the indexer's cache is the only one that
        // rotates.
        std::vector<float> had;
        ds4_fill_hadamard(had, (int) M.h.idx_key_len);
        set(lin.rot, had.data(), had.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) NANO_ABORT("compute failed");

    FILE * f = fopen(p.out_path.c_str(), "wb");
    if (!f) NANO_ABORT("cannot write '%s'", p.out_path.c_str());
    uint32_t n_records = 0;
    fwrite(NTD_MAGIC, 1, sizeof(NTD_MAGIC), f);
    fwrite(&n_records, sizeof(n_records), 1, f);

    // Walk the graph in build order, so repeated names ("norm" appears several
    // times a layer) land in the same order llama.cpp emitted them — which is
    // how dump_inspect.py pairs them up.
    std::vector<float> scratch;
    printf("\n");
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (!(t->flags & GGML_TENSOR_FLAG_OUTPUT) || t->name[0] == '\0') continue;
        write_record(f, t, scratch);
        n_records++;
    }

    fseek(f, (long) sizeof(NTD_MAGIC), SEEK_SET);
    fwrite(&n_records, sizeof(n_records), 1, f);
    fclose(f);

    printf("\nds4-port: wrote %s (%u tensors)\n", p.out_path.c_str(), n_records);
    return 0;
}
