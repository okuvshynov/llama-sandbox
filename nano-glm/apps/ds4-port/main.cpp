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
#include <algorithm>
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
    int32_t     n_ctx     = 4096;  // cache cells; matches dump's default
    int32_t     n_ubatch  = 0;     // 0 = the whole prompt in one chunk
    ds4_stage   stage     = DS4_STAGE_LAYER;
};

static bool parse_stage(const char * s, ds4_stage & out) {
    if (!strcmp(s, "hc_pre"))   { out = DS4_STAGE_HC_PRE;   return true; }
    if (!strcmp(s, "compress")) { out = DS4_STAGE_COMPRESS; return true; }
    if (!strcmp(s, "attn"))     { out = DS4_STAGE_ATTN;     return true; }
    if (!strcmp(s, "layer"))    { out = DS4_STAGE_LAYER;    return true; }
    if (!strcmp(s, "head"))     { out = DS4_STAGE_HEAD;     return true; }
    return false;
}

// "@path" reads the ids from a file. A few thousand ids do not fit comfortably
// on a Windows command line, and the sequences that make a compressed-KV
// comparison mean anything are that long.
static std::string tokens_text(const std::string & arg) {
    if (arg.empty() || arg[0] != '@') {
        return arg;
    }
    FILE * f = fopen(arg.c_str() + 1, "rb");
    if (!f) NANO_ABORT("cannot read token file '%s'", arg.c_str() + 1);
    std::string all;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) all.append(buf, n);
    fclose(f);
    // Strip a UTF-8 BOM. PowerShell's `>` writes one; `strtol` then rejects the
    // first token here while `atoi` silently turns it into 0 in logit-kld's
    // `dump`, so the two tools would disagree about the prompt rather than
    // about the model.
    if (all.size() >= 3 && (unsigned char) all[0] == 0xEF &&
        (unsigned char) all[1] == 0xBB && (unsigned char) all[2] == 0xBF) {
        all.erase(0, 3);
    }
    return all;
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
        while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    }
    return out;
}

// One record, matching dump.cpp's format exactly: name, source type, ne[4],
// element count, then f32 data.
// Returns false if nothing was written, which the caller must not count: the
// record count sits in the header and a reader trusts it, so an over-count
// makes the file unparseable rather than short. That is exactly what happened
// when layer 2 introduced the first F16 named tensors (the compressed keys and
// the masks) and this function skipped them while `main` counted them anyway.
static bool write_record(FILE * f, ggml_tensor * t, std::vector<float> & scratch) {
    const uint64_t n_elem = (uint64_t) ggml_nelements(t);
    scratch.resize((size_t) n_elem);

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, scratch.data(), 0, n_elem * sizeof(float));
    } else if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> tmp((size_t) n_elem);
        ggml_backend_tensor_get(t, tmp.data(), 0, n_elem * sizeof(int32_t));
        for (uint64_t i = 0; i < n_elem; i++) scratch[i] = (float) tmp[i];
    } else {
        // Everything else through ggml's own type traits, as logit-kld's `dump`
        // does — so a reader never has to know what F16 is, and the two files
        // stay comparable.
        const ggml_type_traits * tr = ggml_get_type_traits(t->type);
        if (!tr || !tr->to_float) {
            fprintf(stderr, "ds4-port: %s has type %s with no to_float; skipped\n",
                    t->name, ggml_type_name(t->type));
            return false;
        }
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        tr->to_float(raw.data(), scratch.data(), (int64_t) n_elem);
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
    return true;
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
        else if (!strcmp(a, "-c") && i + 1 < argc) p.n_ctx      = atoi(argv[++i]);
        else if (!strcmp(a, "-ub") && i + 1 < argc) p.n_ubatch  = atoi(argv[++i]);
        else if (!strcmp(a, "--stage") && i + 1 < argc) {
            if (!parse_stage(argv[++i], p.stage)) { p.model_path.clear(); break; }
        }
        else { p.model_path.clear(); break; }
    }
    if (p.model_path.empty() || p.tokens_str.empty()) {
        fprintf(stderr,
            "Usage: ds4-port -m <first-shard.gguf> -T <id,id,...|@file> [-o out.ntd] [-t threads] [-L layers]\n"
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

    const std::vector<int32_t> tokens = parse_tokens(tokens_text(p.tokens_str));
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

    // The cache is sized to a *context*, not to a chunk. That matters for more
    // than capacity: `get_n_kv` caps its padding at the cache size, so a cache
    // sized to the prompt would quietly hand the graph an unpadded window and
    // stop the key views matching the reference shapes.
    if (p.n_ctx < n_tokens) NANO_ABORT("-c %d is smaller than the %d tokens", p.n_ctx, n_tokens);
    ds4_cache C;
    ds4_cache_init(C, M.h, (uint32_t) p.n_ctx);

    FILE * f = fopen(p.out_path.c_str(), "wb");
    if (!f) NANO_ABORT("cannot write '%s'", p.out_path.c_str());
    uint32_t n_records = 0;
    fwrite(NTD_MAGIC, 1, sizeof(NTD_MAGIC), f);
    fwrite(&n_records, sizeof(n_records), 1, f);

    std::vector<float> scratch;
    const int32_t chunk_size = p.n_ubatch > 0 ? p.n_ubatch : n_tokens;

    // One chunk: build a graph for (n_past, n_tok), run it, append every named
    // tensor. The cache lives outside this and is what carries state forward,
    // so calling it twice is the whole of multi-chunk prefill.
    //
    // The graph is rebuilt per chunk rather than reused: its shapes depend on
    // both n_tok and the padded window, and llama.cpp rebuilds for the same
    // reason. A production engine would cache by (n_tok, n_kv) as glm-dsa's
    // `eval_chunk` does; a porting harness gains nothing from it.
    auto run_chunk = [&](int32_t n_past, const int32_t * chunk, int32_t n_tok) {
        const size_t n_nodes_max = 32768;
        std::vector<uint8_t> meta(ggml_tensor_overhead() * n_nodes_max +
                                  ggml_graph_overhead_custom(n_nodes_max, false));
        ggml_init_params ip = { meta.size(), meta.data(), /*no_alloc =*/ true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) NANO_ABORT("no graph context");

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_nodes_max, false);

        auto new_input = [&](ggml_type type, int64_t ne0, int64_t ne1, const char * name) {
            ggml_tensor * t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
            ggml_set_name(t, name);
            ggml_set_input(t);
            return t;
        };

        // llama.cpp's `get_n_kv`: at least 256 cells, rounded up to 256, capped
        // by the cache. Matching it is not cosmetic — it is what makes the key
        // views and masks the same shape as the reference's.
        auto pad_n_kv = [](int32_t used, uint32_t size) {
            return (int32_t) std::min<int64_t>(size, std::max<int64_t>(256, GGML_PAD(used, 256)));
        };
        const int32_t n_used   = n_past + n_tok;
        const int32_t n_kv_raw = pad_n_kv(n_used, C.kv_size);
        // Zero until a block has actually closed, and the padding floor must not
        // hide that: llama.cpp leaves the compressed mask null while no block
        // exists, and the layer then runs plain raw attention. Padding first
        // would make every layer take the compressed path from token 0.
        const int32_t n_kv_csa = n_used / 4   > 0 ? pad_n_kv(n_used / 4,   C.csa_size) : 0;
        const int32_t n_kv_hca = n_used / 128 > 0 ? pad_n_kv(n_used / 128, C.hca_size) : 0;

        // The reserve count is set by the *configured* chunk size, not by this
        // chunk's length, because that is what llama.cpp reserves the graph for.
        const int32_t reserve = (chunk_size + 3) / 4;
        const ds4_comp_plan p_csa = ds4_plan_comp(4,   true,  n_past, n_tok, n_kv_csa,
                                                  (int32_t) C.csa_size, reserve);
        const ds4_comp_plan p_hca = ds4_plan_comp(128, false, n_past, n_tok, n_kv_hca,
                                                  (int32_t) C.hca_size, 0);

        std::vector<ggml_fp16_t> raw_mask;
        ds4_plan_raw_mask(raw_mask, n_kv_raw, n_past, n_tok, (int32_t) M.h.sliding_window);

        std::vector<int64_t> k_idxs((size_t) n_tok);
        for (int32_t i = 0; i < n_tok; i++) k_idxs[i] = n_past + i;

        ggml_tensor * inp_tokens = new_input(GGML_TYPE_I32, n_tok, 1, "inp_tokens");
        ggml_tensor * inp_pos    = new_input(GGML_TYPE_I32, n_tok, 1, "inp_pos");

        auto declare = [&](ds4_comp_inputs & d, const ds4_comp_plan & pl, const char * tag) {
            char nm[64];
            auto nmk = [&](const char * suffix) {
                snprintf(nm, sizeof(nm), "%s_%s", tag, suffix);
                return nm;
            };
            auto i32v = [&](const std::vector<int32_t> & v, const char * suf) {
                return v.empty() ? (ggml_tensor *) nullptr
                                 : new_input(GGML_TYPE_I32, (int64_t) v.size(), 1, nmk(suf));
            };
            auto i64v = [&](const std::vector<int64_t> & v, const char * suf) {
                return v.empty() ? (ggml_tensor *) nullptr
                                 : new_input(GGML_TYPE_I64, (int64_t) v.size(), 1, nmk(suf));
            };
            d.read_idxs   = i32v(pl.read_idxs,   "read_idxs");
            d.comp_pos    = i32v(pl.write_pos,   "comp_pos");
            d.ape_pos     = i32v(pl.ape_pos,     "ape_pos");
            d.write_idxs  = i64v(pl.write_idxs,  "write_idxs");
            d.persist_src = i32v(pl.persist_src, "persist_src");
            d.persist_dst = i64v(pl.persist_dst, "persist_dst");
            // No mask when nothing has been compressed yet — that is what
            // tells the graph to skip the compressed half entirely.
            d.mask        = pl.n_kv > 0
                          ? new_input(GGML_TYPE_F16, pl.n_kv, n_tok, nmk("kq_mask"))
                          : nullptr;
        };

        ds4_layer_inputs lin;
        declare(lin.csa, p_csa, "csa");
        declare(lin.lid, p_csa, "lid");
        declare(lin.hca, p_hca, "hca");
        lin.rot    = new_input(GGML_TYPE_F32, M.h.idx_key_len, M.h.idx_key_len, "lid_k_rot");
        lin.k_idxs = new_input(GGML_TYPE_I64, n_tok, 1, "k_idxs");

        // F16: ggml_flash_attn_ext requires it, where the explicit path wants F32.
        ggml_tensor * kq_mask = new_input(GGML_TYPE_F16, n_kv_raw, n_tok, "kq_mask");

        // llama.cpp asks for logits at the last position only, so the head runs
        // on one row. A head over every position would differ in shape.
        ggml_tensor * out_ids = nullptr;
        if (p.stage == DS4_STAGE_HEAD) {
            out_ids = new_input(GGML_TYPE_I32, 1, 1, "inp_out_ids");
        }

        ggml_tensor * out = ds4_build_graph(ctx, gf, M, C, inp_tokens, inp_pos, kq_mask, lin,
                                            out_ids, n_tok, p.stage, (uint32_t) p.n_layers);
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) NANO_ABORT("graph alloc failed");

        // An input the graph never referenced has no buffer, because gallocr
        // only allocates what the graph reaches. That is not a hazard to route
        // around, it is the correct question to ask: a chunk that closes no
        // block does not rotate anything, so `lid_k_rot` is genuinely unused
        // and filling it would be meaningless. An input the graph *does* use is
        // always allocated, so this cannot silently skip something needed.
        auto set = [](ggml_tensor * t, const void * d, size_t n) {
            if (t && t->buffer) ggml_backend_tensor_set(t, d, 0, n);
        };

        set(inp_tokens, chunk, n_tok * sizeof(int32_t));
        {
            std::vector<int32_t> pos((size_t) n_tok);
            for (int32_t i = 0; i < n_tok; i++) pos[i] = n_past + i;
            set(inp_pos, pos.data(), pos.size() * sizeof(int32_t));
        }
        set(kq_mask, raw_mask.data(), raw_mask.size() * sizeof(ggml_fp16_t));
        set(lin.k_idxs, k_idxs.data(), k_idxs.size() * sizeof(int64_t));
        auto upload = [&](const ds4_comp_inputs & d, const ds4_comp_plan & pl) {
            set(d.read_idxs,   pl.read_idxs.data(),   pl.read_idxs.size()   * sizeof(int32_t));
            set(d.comp_pos,    pl.write_pos.data(),   pl.write_pos.size()   * sizeof(int32_t));
            set(d.ape_pos,     pl.ape_pos.data(),     pl.ape_pos.size()     * sizeof(int32_t));
            set(d.write_idxs,  pl.write_idxs.data(),  pl.write_idxs.size()  * sizeof(int64_t));
            set(d.persist_src, pl.persist_src.data(), pl.persist_src.size() * sizeof(int32_t));
            set(d.persist_dst, pl.persist_dst.data(), pl.persist_dst.size() * sizeof(int64_t));
            set(d.mask,        pl.mask.data(),        pl.mask.size()        * sizeof(ggml_fp16_t));
        };
        upload(lin.csa, p_csa);
        upload(lin.lid, p_csa);
        upload(lin.hca, p_hca);

        // Order 128, not d_key: the indexer's cache is the only one that rotates.
        std::vector<float> had;
        ds4_fill_hadamard(had, (int) M.h.idx_key_len);
        set(lin.rot, had.data(), had.size() * sizeof(float));

        if (out_ids) {
            const int32_t last = n_tok - 1;
            set(out_ids, &last, sizeof(int32_t));
        }

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            NANO_ABORT("compute failed");
        }

        // Walk in build order, so repeated names ("norm" appears several times a
        // layer) land in the order llama.cpp emitted them — which is how
        // dump_inspect.py pairs them up, chunk by chunk as well as within one.
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (!(t->flags & GGML_TENSOR_FLAG_OUTPUT) || t->name[0] == '\0') continue;
            if (write_record(f, t, scratch)) {
                n_records++;
            }
        }

        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    };

    for (int32_t off = 0; off < n_tokens; off += chunk_size) {
        const int32_t n_tok = std::min(chunk_size, n_tokens - off);
        fprintf(stderr, "ds4-port: chunk at %d, %d tokens\n", off, n_tok);
        printf("\n");
        run_chunk(off, tokens.data() + off, n_tok);
    }

    fseek(f, (long) sizeof(NTD_MAGIC), SEEK_SET);
    fwrite(&n_records, sizeof(n_records), 1, f);
    fclose(f);

    ds4_cache_free(C);
    printf("\nds4-port: wrote %s (%u tensors)\n", p.out_path.c_str(), n_records);
    return 0;
}
