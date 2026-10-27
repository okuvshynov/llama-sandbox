#include "model.h"

#include "ggml-cpu.h"
#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <set>
#include <stdexcept>

static std::runtime_error err(const std::string& s) { return std::runtime_error(s); }

std::string fmt_bytes(double b) {
  char s[32];
  if (b >= 1 << 30) snprintf(s, sizeof s, "%.2f GiB", b / (1 << 30));
  else snprintf(s, sizeof s, "%.1f MiB", b / (1 << 20));
  return s;
}

// ------------------------------------------------------------------ GGUF metadata helpers

namespace {
struct Meta {
  gguf_context* g;
  std::string arch;

  int64_t id(const std::string& key) const {
    int64_t i = gguf_find_key(g, key.c_str());
    if (i < 0) throw err("missing GGUF key " + key);
    return i;
  }
  std::string a(const char* k) const { return arch + "." + k; }

  int64_t i64(const std::string& key) const {
    int64_t i = id(key);
    switch (gguf_get_kv_type(g, i)) {
      case GGUF_TYPE_UINT8: return gguf_get_val_u8(g, i);
      case GGUF_TYPE_INT32: return gguf_get_val_i32(g, i);
      case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, i);
      case GGUF_TYPE_INT64: return gguf_get_val_i64(g, i);
      case GGUF_TYPE_UINT64: return (int64_t)gguf_get_val_u64(g, i);
      case GGUF_TYPE_BOOL: return gguf_get_val_bool(g, i);
      default: throw err("key " + key + " is not an integer");
    }
  }
  float f32(const std::string& key) const {
    int64_t i = id(key);
    if (gguf_get_kv_type(g, i) != GGUF_TYPE_FLOAT32) throw err("key " + key + " is not f32");
    return gguf_get_val_f32(g, i);
  }
  template <class T>
  std::vector<T> arr(const std::string& key) const {
    int64_t i = id(key);
    if (gguf_get_kv_type(g, i) != GGUF_TYPE_ARRAY) throw err("key " + key + " is not an array");
    const size_t n = gguf_get_arr_n(g, i);
    const void* d = gguf_get_arr_data(g, i);
    std::vector<T> out(n);
    for (size_t j = 0; j < n; j++) {
      switch (gguf_get_arr_type(g, i)) {
        case GGUF_TYPE_INT32: out[j] = (T)((const int32_t*)d)[j]; break;
        case GGUF_TYPE_UINT32: out[j] = (T)((const uint32_t*)d)[j]; break;
        case GGUF_TYPE_INT64: out[j] = (T)((const int64_t*)d)[j]; break;
        case GGUF_TYPE_UINT64: out[j] = (T)((const uint64_t*)d)[j]; break;
        case GGUF_TYPE_FLOAT32: out[j] = (T)((const float*)d)[j]; break;
        default: throw err("array " + key + " has an unsupported element type");
      }
    }
    return out;
  }
};

bool contains(const std::vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }
}  // namespace

static Hparams read_hparams(gguf_context* g) {
  Meta m{g, ""};
  int64_t ai = gguf_find_key(g, "general.architecture");
  if (ai < 0) throw err("not a model GGUF (no general.architecture)");
  m.arch = gguf_get_val_str(g, ai);
  if (m.arch != "deepseek41") throw err("architecture is " + m.arch + ", expected deepseek41");

  Hparams h;
  h.n_layer = (int)m.i64(m.a("block_count"));
  h.n_embd = (int)m.i64(m.a("embedding_length"));
  h.n_vocab = (int)gguf_get_arr_n(g, m.id("tokenizer.ggml.tokens"));
  h.n_head = (int)m.i64(m.a("attention.head_count"));
  h.head_dim = (int)m.i64(m.a("attention.key_length"));
  h.rope_dim = (int)m.i64(m.a("rope.dimension_count"));
  h.q_lora = (int)m.i64(m.a("attention.q_lora_rank"));
  h.o_groups = (int)m.i64(m.a("attention.output_group_count"));
  h.o_lora = (int)m.i64(m.a("attention.output_lora_rank"));
  h.window = (int)m.i64(m.a("attention.sliding_window"));
  h.rms_eps = m.f32(m.a("attention.layer_norm_rms_epsilon"));
  h.rope_base = m.f32(m.a("rope.freq_base"));
  h.compress_rope_base = m.f32(m.a("attention.compress_rope_freq_base"));
  h.yarn_factor = m.f32(m.a("rope.scaling.factor"));
  h.yarn_orig_ctx = (int)m.i64(m.a("rope.scaling.original_context_length"));
  h.yarn_beta_fast = m.f32(m.a("rope.scaling.yarn_beta_fast"));
  h.yarn_beta_slow = m.f32(m.a("rope.scaling.yarn_beta_slow"));
  h.compress_ratios = m.arr<int>(m.a("attention.compress_ratios"));
  h.kv_source_layers = m.arr<int>(m.a("attention.kv_source_layers"));
  h.index_source_layers = m.arr<int>(m.a("attention.indexer.source_layers"));
  h.cand_source = (int)m.i64(m.a("attention.indexer.candidate_source_layer"));
  h.cand_topk_blocks = (int)m.i64(m.a("attention.indexer.candidate_top_k_blocks"));
  h.cand_block = (int)m.i64(m.a("attention.indexer.candidate_block_size"));
  h.idx_heads = (int)m.i64(m.a("attention.indexer.head_count"));
  h.idx_dim = (int)m.i64(m.a("attention.indexer.key_length"));
  h.idx_topk = (int)m.i64(m.a("attention.indexer.top_k"));
  h.n_expert = (int)m.i64(m.a("expert_count"));
  h.n_expert_used = (int)m.i64(m.a("expert_used_count"));
  h.n_ff_exp = (int)m.i64(m.a("expert_feed_forward_length"));
  h.gating_func = (int)m.i64(m.a("expert_gating_func"));
  h.expert_weights_scale = m.f32(m.a("expert_weights_scale"));
  h.expert_weights_norm = m.i64(m.a("expert_weights_norm")) != 0;
  h.swiglu_clamp = m.arr<float>(m.a("swiglu_clamp_exp"));
  h.swiglu_clamp_sh = m.arr<float>(m.a("swiglu_clamp_shexp"));
  h.hc_mult = (int)m.i64(m.a("hyper_connection.count"));
  h.hc_iters = (int)m.i64(m.a("hyper_connection.sinkhorn_iterations"));
  h.hc_eps = m.f32(m.a("hyper_connection.epsilon"));
  h.engram_layers = m.arr<int>(m.a("ple.layers"));
  h.engram_ngram = (int)m.i64(m.a("ple.ngram_size"));
  h.engram_heads = (int)m.i64(m.a("ple.heads_per_ngram"));
  h.engram_dim = (int)m.i64(m.a("embedding_length_per_layer_input"));
  h.engram_pad = (int)m.i64(m.a("engram.pad_token_id"));
  h.engram_mult = m.arr<uint64_t>(m.a("ple.layer_multipliers"));
  h.engram_offsets = m.arr<uint64_t>(m.a("ple.head_offsets"));
  h.engram_primes = m.arr<uint64_t>(m.a("ple.head_vocab_sizes"));
  h.engram_token_map = m.arr<int32_t>(m.a("engram.token_map"));

  // invariants the rest of the code relies on
  auto need = [](bool ok, const char* what) { if (!ok) throw err(std::string("hparams: ") + what); };
  need((int)h.compress_ratios.size() >= h.n_layer, "compress_ratios shorter than block_count");
  h.compress_ratios.resize(h.n_layer);  // drop the MTP entries: the GGUF has no MTP layers
  need((int)h.swiglu_clamp.size() == h.n_layer && (int)h.swiglu_clamp_sh.size() == h.n_layer, "swiglu clamps");
  need(h.n_head * h.head_dim % h.o_groups == 0, "heads split into o_groups");
  need(h.engram_mult.size() == h.engram_layers.size() * h.engram_ngram, "engram multipliers");
  need(h.engram_offsets.size() == h.engram_layers.size() * h.engram_cols(), "engram offsets");
  need(h.engram_primes.size() == h.engram_offsets.size(), "engram primes");
  need((int)h.engram_token_map.size() == h.n_vocab, "engram token map covers the vocab");
  for (int L : h.kv_source_layers) need(h.compress_ratios[L] > 0, "kv source with ratio 0");
  for (int L : h.index_source_layers) need(h.compress_ratios[L] > 0, "index source with ratio 0");
  return h;
}

static LayerKind layer_kind(const Hparams& h, int L) {
  LayerKind k;
  k.ratio = h.compress_ratios[L];
  k.kv_source = contains(h.kv_source_layers, L);
  k.index_source = contains(h.index_source_layers, L);
  k.cand_source = L == h.cand_source;
  k.uses_cands = k.index_source && h.cand_source >= 0 && h.cand_source < L;
  if (k.ratio > 0) {
    // layers read what the most recent source (in execution order) published
    for (int s : h.kv_source_layers) if (s <= L) k.kv_from = s;
    for (int s : h.index_source_layers) if (s <= L) k.idx_from = s;
    if (k.kv_from < 0 || k.idx_from < 0) throw err("layer " + std::to_string(L) + ": compressed attention before any source");
    if (h.compress_ratios[k.kv_from] != k.ratio) throw err("layer " + std::to_string(L) + ": ratio differs from its KV source");
  }
  for (size_t e = 0; e < h.engram_layers.size(); e++) if (h.engram_layers[e] == L) k.engram = (int)e;
  return k;
}

// ------------------------------------------------------------------ loading

Model::~Model() {
  for (auto& s : shards) {
    if (s.buf) ggml_backend_buffer_free(s.buf);
    if (s.ctx) ggml_free(s.ctx);
    if (s.g) gguf_free(s.g);
    if (s.map) munmap(s.map, s.size);
    if (s.fd >= 0) close(s.fd);
  }
}

ggml_tensor* Model::get(const std::string& name) const {
  auto it = by_name.find(name);
  return it == by_name.end() ? nullptr : it->second;
}

void Model::load(const std::string& first, const std::vector<int>& want_layers) {
  // split naming: <prefix>-00001-of-000NN.gguf
  auto pos = first.rfind("-00001-of-");
  if (pos == std::string::npos) throw err("expected the first shard (…-00001-of-000NN.gguf): " + first);
  const int n_split = atoi(first.c_str() + pos + 10);
  for (int i = 1; i <= n_split; i++) {
    char suf[32];
    snprintf(suf, sizeof suf, "-%05d-of-%05d.gguf", i, n_split);
    shard_paths.push_back(first.substr(0, pos) + suf);
  }

  for (const auto& path : shard_paths) {
    Shard s;
    gguf_init_params p = {/*no_alloc*/ true, &s.ctx};
    s.g = gguf_init_from_file(path.c_str(), p);
    if (!s.g) throw err("cannot read GGUF " + path);
    s.fd = open(path.c_str(), O_RDONLY);
    struct stat st;
    if (s.fd < 0 || fstat(s.fd, &st) != 0) throw err("cannot open " + path);
    s.size = (size_t)st.st_size;
    s.map = mmap(nullptr, s.size, PROT_READ, MAP_SHARED, s.fd, 0);
    if (s.map == MAP_FAILED) throw err("mmap failed: " + path);
    // ggml's CPU buffer wants a non-const pointer; the weights are only ever read
    s.buf = ggml_backend_cpu_buffer_from_ptr(s.map, s.size);
    // "weights": the scheduler then runs ops on these where they live, instead of copying them
    ggml_backend_buffer_set_usage(s.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const size_t data0 = gguf_get_data_offset(s.g);
    for (int64_t i = 0; i < gguf_get_n_tensors(s.g); i++) {
      const char* name = gguf_get_tensor_name(s.g, i);
      ggml_tensor* t = ggml_get_tensor(s.ctx, name);
      const size_t off = data0 + gguf_get_tensor_offset(s.g, i);
      if (off + ggml_nbytes(t) > s.size) throw err(std::string("tensor past end of shard: ") + name);
      if (ggml_backend_tensor_alloc(s.buf, t, (char*)s.map + off) != GGML_STATUS_SUCCESS)
        throw err(std::string("cannot bind tensor ") + name);
      if (!by_name.emplace(name, t).second) throw err(std::string("tensor in two shards: ") + name);
    }
    shards.push_back(s);
  }
  hp = read_hparams(shards[0].g);

  std::set<std::string> claimed;
  // Look up one tensor and check it is exactly what we expect.
  auto req = [&](const std::string& name, ggml_type type, std::vector<int64_t> ne) -> ggml_tensor* {
    ggml_tensor* t = get(name);
    if (!t) throw err("missing tensor " + name);
    if (t->type != type)
      throw err(name + ": type " + ggml_type_name(t->type) + ", expected " + ggml_type_name(type));
    ne.resize(GGML_MAX_DIMS, 1);
    for (int d = 0; d < GGML_MAX_DIMS; d++)
      if (t->ne[d] != ne[d]) {
        char s[256];
        snprintf(s, sizeof s, "%s: ne = [%" PRId64 ", %" PRId64 ", %" PRId64 "], expected [%" PRId64 ", %" PRId64 ", %" PRId64 "]",
                 name.c_str(), t->ne[0], t->ne[1], t->ne[2], ne[0], ne[1], ne[2]);
        throw err(s);
      }
    claimed.insert(name);
    return t;
  };
  const auto& h = hp;
  const int64_t E = h.n_embd, F = h.n_ff_exp, X = h.n_expert, HD = h.head_dim;
  const auto BF = GGML_TYPE_BF16, F32 = GGML_TYPE_F32;

  tok_embd = req("token_embd.weight", BF, {E, h.n_vocab});
  output = req("output.weight", BF, {E, h.n_vocab});
  output_norm = req("output_norm.weight", F32, {E});
  {
    int64_t rows = 0;
    for (size_t c = 0; c < h.engram_offsets.size(); c++)
      rows = std::max<int64_t>(rows, (int64_t)(h.engram_offsets[c] + h.engram_primes[c]));
    engram_table = req("per_layer_token_embd.weight", GGML_TYPE_MXFP8, {h.engram_dim, rows});
  }

  layers.assign(h.n_layer, Layer{});
  std::vector<int> sel = want_layers;
  if (sel.empty()) for (int L = 0; L < h.n_layer; L++) sel.push_back(L);
  for (int L : sel) {
    if (L < 0 || L >= h.n_layer) throw err("layer out of range: " + std::to_string(L));
    Layer& l = layers[L];
    l.kind = layer_kind(h, L);
    auto n = [L](const char* s) { return "blk." + std::to_string(L) + "." + s; };
    l.attn_norm = req(n("attn_norm.weight"), F32, {E});
    l.q_a = req(n("attn_q_a.weight"), BF, {E, h.q_lora});
    l.q_a_norm = req(n("attn_q_a_norm.weight"), F32, {h.q_lora});
    l.q_b = req(n("attn_q_b.weight"), BF, {h.q_lora, h.n_head * HD});
    l.kv = req(n("attn_kv.weight"), BF, {E, HD});
    l.kv_norm = req(n("attn_kv_a_norm.weight"), F32, {HD});
    l.out_a = req(n("attn_output_a.weight"), BF, {h.n_head * HD / h.o_groups, h.o_groups * h.o_lora});
    l.out_b = req(n("attn_output_b.weight"), BF, {h.o_groups * h.o_lora, E});
    l.sinks = req(n("attn_sinks.weight"), F32, {h.n_head});
    if (l.kind.kv_source) {
      l.cmp_kv = req(n("attn_compressor_kv.weight"), BF, {E, HD});
      if (l.kind.ratio > 1) l.cmp_gate = req(n("attn_compressor_gate.weight"), BF, {E, HD});
      l.cmp_norm = req(n("attn_compressor_norm.weight"), F32, {HD});
      l.idx_k = req(n("indexer.attn_k.weight"), BF, {HD, h.idx_dim});
      l.idx_k_norm = req(n("indexer.k_norm.weight"), F32, {h.idx_dim});
    }
    if (l.kind.index_source) {
      l.idx_q_b = req(n("indexer.attn_q_b.weight"), BF, {h.q_lora, h.idx_heads * h.idx_dim});
      l.idx_proj = req(n("indexer.proj.weight"), BF, {E, h.idx_heads});
    }
    l.ffn_norm = req(n("ffn_norm.weight"), F32, {E});
    l.gate_inp = req(n("ffn_gate_inp.weight"), F32, {E, X});
    l.gate_bias = req(n("exp_probs_b.bias"), F32, {X});
    l.gate_bias_vl = req(n("exp_probs_b_vl.bias"), F32, {X});
    l.gate_exps = req(n("ffn_gate_exps.weight"), GGML_TYPE_MXFP4, {E, F, X});
    l.up_exps = req(n("ffn_up_exps.weight"), GGML_TYPE_MXFP4, {E, F, X});
    l.down_exps = req(n("ffn_down_exps.weight"), GGML_TYPE_MXFP4, {F, E, X});
    l.gate_sh = req(n("ffn_gate_shexp.weight"), BF, {E, F});
    l.up_sh = req(n("ffn_up_shexp.weight"), BF, {E, F});
    l.down_sh = req(n("ffn_down_shexp.weight"), BF, {F, E});
    const int64_t mix = (2 + h.hc_mult) * h.hc_mult;
    l.hc_attn_fn = req(n("hc_attn_fn.weight"), F32, {h.hc_mult * E, mix});
    l.hc_attn_base = req(n("hc_attn_base.weight"), F32, {mix});
    l.hc_attn_scale = req(n("hc_attn_scale.weight"), F32, {3});
    l.hc_ffn_fn = req(n("hc_ffn_fn.weight"), F32, {h.hc_mult * E, mix});
    l.hc_ffn_base = req(n("hc_ffn_base.weight"), F32, {mix});
    l.hc_ffn_scale = req(n("hc_ffn_scale.weight"), F32, {3});
    if (l.kind.engram >= 0) {
      l.engram_kv = req(n("engram_kv.weight"), BF, {h.engram_cols() * h.engram_dim, E * (h.hc_mult + 1)});
      l.engram_gate = req(n("engram_gate.weight"), F32, {E, h.hc_mult});
    }
    l.loaded = true;
  }

  // Every tensor of a loaded layer (and every global one) must have been claimed: an unexpected
  // tensor means the file has structure this code does not know about.
  std::set<int> loaded(sel.begin(), sel.end());
  for (const auto& [name, t] : by_name) {
    int L = -1;
    if (sscanf(name.c_str(), "blk.%d.", &L) == 1) {
      if (L < 0 || L >= h.n_layer) throw err("tensor for a layer that does not exist: " + name);
      if (!loaded.count(L)) continue;
    }
    if (!claimed.count(name)) throw err("unexpected tensor (not in the schema): " + name);
  }
}
