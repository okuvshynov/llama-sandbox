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
};

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
        else { p.model_path.clear(); break; }
    }
    if (p.model_path.empty() || p.tokens_str.empty()) {
        fprintf(stderr,
            "Usage: ds4-port -m <first-shard.gguf> -T <id,id,...> [-o out.ntd] [-t threads]\n"
            "  Runs the deepseek4 trunk as far as it is ported and dumps every\n"
            "  named tensor, for comparison against logit-kld's `dump` of\n"
            "  llama.cpp. Token ids, not text: that keeps the tokenizer out of\n"
            "  the comparison.\n");
        return 1;
    }

    const std::vector<int32_t> tokens = parse_tokens(p.tokens_str);
    const int32_t n_tokens = (int32_t) tokens.size();

    ds4_model M;
    ds4_load_model(M, p.model_path);
    fprintf(stderr, "ds4-port: %s | %s\n", M.desc.c_str(), nano_build_line().c_str());
    fprintf(stderr, "ds4-port: %d tokens, %d threads\n", n_tokens, p.n_threads);

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

    // No rotation input: this model's 512-wide attention caches are F16, and
    // llama.cpp only builds a Hadamard rotation for a quantized cache. One
    // comes back when the lightning-indexer layers do, at order 128.
    ggml_tensor * out = ds4_build_graph(ctx, gf, M, inp_tokens, inp_pos, kq_mask,
                                        n_tokens, DS4_STAGE_LAYER);
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
