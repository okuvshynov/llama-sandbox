// The own decode runtime (PLAN.md section 3, option 2): one token per step through our HIP
// kernels (kernels.h, fp8_gemv.h, fp4_moe.h). The dense part runs on one die (opt.gpu); the
// routed experts run where they live: on any die that holds them in an expert pool
// (expert_pool.h, --placement) or on the CPU (cpu_moe.h, AVX-512, weights read in place from
// the mmap), each layer's GPU and CPU parts at the same time.
//
// Self-contained: it reads the model's host tensors (model.h), uploads the small weights itself,
// and keeps the sequence state (window rings, compressor groups, compressed KV, index keys) in
// its own device buffers; every token, prompt included, runs here. Every layer can be checked
// teacher-forced against the reference (ds-test --runtime).
// Weights: the fp8 linears from fp8pack's file (exact fp8 x fp8 GEMV); the small f32/bf16
// tensors (norms, hc, router, compressor, indexer projections, head) uploaded from the model.
// Numerics are the reference's: fp32 everywhere, fake quantization where it has it.
#pragma once

#include "forward.h"  // Taps, engram_rows
#include "dspark.h"
#include "expert_cache.h"
#include "expert_loader.h"
#include "expert_pool.h"
#include "fp4_moe.h"
#include "kernels.h"
#include "model.h"

#include <hip/hip_runtime.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

class CpuMoeKernel;  // src/cpu_moe.h: the routed experts on the CPU, exact fp4 x fp8

struct RtOptions {
  int gpu = 0;         // the die for the dense part and the state (ROCm device index)
  int n_threads = 16;  // CPU expert kernel
  int max_ctx = 4096;  // capacity of the compressed caches
  int prefill_chunk = 512;  // prompt tokens per forward (1 .. Runtime::kChunk)
  bool prefill_overlap = true;  // a prompt chunk in two halves: CPU experts overlap GPU work
  // routed experts on GPUs (expert_pool.h): a ranked placement file fills these dies' free
  // memory, keeping `reserve_gb` free on each; empty = every routed expert on the CPU
  std::string placement;
  std::vector<int> moe_dies = {0, 1, 2, 3};
  double reserve_gb = 0.75;
  // the DSpark draft head's GGUF (dspark.h); empty = none
  std::string dspark;
  // load test (README step 30): >= 0 keeps kLoadTestSlots spare slots per die and loads random
  // experts into them at this many loads/s in all (0: spare slots, no loads), in the background,
  // as an adaptive cache would; nothing reads them, so the output must not change
  double load_test = -1;
  // placement-invariant expert numerics (canon_dot.h; PLAN.md "Adaptive expert cache"): every
  // routed expert gives the same bits on any die or the CPU and they are summed in router-rank
  // order, so the output does not depend on the placement (or on a cache's contents); slower
  // expert kernels (README step 28)
  bool canon = false;
  // adaptive expert cache (expert_cache.h): > 0 = load rate in experts per second, rebalancing
  // every cache_period tokens by use over the last cache_window; 0 = the static placement
  double cache_loads = 0;
  int cache_period = 16, cache_window = 256;
  bool cache_wait = false;  // wait at startup until the cache can load (expert_cache.h)
  bool load_pinned = false;  // the load test's path: pinned staging instead of direct (expert_loader.h)
};

class Runtime {
 public:
  // fp8_path is fp8pack's output for the same model
  Runtime(const Model& m, RtOptions opt, const std::string& fp8_path);
  ~Runtime();

  // one step for `token` at position n_past(): returns the logits (empty when
  // want_logits is false: prompt tokens before the last skip the 1.3 GB head)
  std::vector<float> decode(int32_t token, bool want_logits = true);
  // the prompt: all but its last token in chunks of up to kChunk (forward_chunk, no head), then
  // decode() for the last; its logits
  std::vector<float> prefill(const std::vector<int32_t>& tokens);

  // teacher-forced single layer (tests): begin_step(token), then run_layer for L = 0.. in
  // order (consumers read what their source layer wrote), then end_step(). Inputs and taps as
  // Engine::run_layer (h_in is the stream before the engram on engram layers).
  void begin_step(int32_t token);
  Taps run_layer(int L, const std::vector<float>& h_in, const std::vector<float>& pre_in);
  void end_step();
  int n_past() const { return n_past_; }

  // DSpark draft (RtOptions::dspark): after a decode step produced `anchor` (the next token),
  // the draft's block_size guesses for the tokens after it. Does not change the sequence state.
  bool has_draft() const { return dspark != nullptr; }
  int draft_block() const { return dspark ? dspark->block_size : 0; }

  // Speculative verification (PLAN.md "batched verification"; phase 1: token by token through
  // decode()). verify(chunk) runs the chunk at positions n_past()..; returns the greedy next
  // token after each. commit(n_keep) then keeps the first n_keep and discards the rest: the
  // window ring slots the discarded positions overwrote are restored, the sequence is truncated,
  // and the draft is seeded with the kept positions only. The compressor state needs nothing:
  // it is keyed by position and only read after the token completing it, which the real tokens
  // rewrite.
  // head = false: no logits (prompt chunks), returns an empty vector
  std::vector<int32_t> verify(const std::vector<int32_t>& chunk, bool head = true);
  // most tokens one forward runs at once: a prompt chunk (prefill); a verify chunk is at most
  // DSpark::kMaxChunk. Per-token buffers are sized by it.
  static constexpr int kChunk = 512;
  void commit(int n_keep);
  DraftResult draft(int32_t anchor, bool keep_logits = false);
  void draft_log_hidden(const std::string& path) {
    if (!dspark) return;
    dspark->hidden_log = fopen(path.c_str(), "wb");
    dspark->debug_log = fopen((path + ".debug").c_str(), "wb");
  }

  // time spent per decode step (host wall clock), split at the CPU-expert waits
  struct Times { double total_ms = 0, cpu_moe_ms = 0, wait_ms = 0; int gpu_experts = 0, cpu_experts = 0; } last;
  long total_gpu_experts = 0, total_cpu_experts = 0;  // over the run (hit rate)
  // profile = true: GPU events between groups of launches; per group, the GPU time of the
  // last step (gaps included: a group charged with idle time was waiting on the host)
  bool profile = false;
  // routing log (scripts/route_stats.py reads it): opens `path` and appends one record per token
  // forwarded (prompt tokens and rejected drafts included): its position, id, whether it is a
  // prompt token, the forward it ran in, and per layer the router's expert ids and weights.
  // Format in rt.cpp (Runtime::log_routing).
  void log_routing(const std::string& path);
  std::map<std::string, double> last_groups;

 private:
  friend class DSpark;
  std::unique_ptr<DSpark> dspark;
  struct Fp8W { uint8_t* q = nullptr; uint8_t* s = nullptr; int N = 0, K = 0; };
  struct LayerW {
    Fp8W q_a, q_b, kv, out_a, out_b, idx_q_b, gate_sh, up_sh, down_sh, engram_kv;
    const float *attn_norm, *q_a_norm, *kv_norm, *sinks, *ffn_norm, *gate_inp, *gate_bias;
    const float *hc_attn_fn, *hc_attn_base, *hc_attn_scale, *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
    const float *cmp_kv = nullptr, *cmp_gate = nullptr, *cmp_norm = nullptr, *idx_k = nullptr, *idx_k_norm = nullptr;
    const uint16_t* idx_proj = nullptr;  // bf16
    const float* engram_gate = nullptr;
    const float* freqs;  // [rope_dim / 2] this layer's rotary frequencies
  };

  const Model& m;
  const Hparams& hp;
  RtOptions opt;
  std::vector<void*> owned;  // device copies of the small weights
  const uint16_t* out_w = nullptr;  // head, bf16
  const float* out_norm = nullptr;
  // the sequence state, per loaded layer (device)
  struct LayerState {
    float* win = nullptr;                          // window KV ring [window][head_dim]
    float *cmp_kv = nullptr, *cmp_score = nullptr; // compressor partial group [ratio][head_dim] (ratio > 1)
    float *comp = nullptr, *idx_k = nullptr;       // compressed KV, index keys [max_ctx / ratio][...] (KV sources)
  };
  std::vector<LayerState> S;
  int n_past_ = 0;
  std::vector<int32_t> tokens_;  // all tokens so far
  hipStream_t s;
  std::vector<LayerW> W;
  void* fp8_buf = nullptr;  // all fp8 weights + scales
  float* freqs_buf = nullptr;
  std::unique_ptr<CpuMoeKernel> cpu_moe;
  // Routed experts on GPUs: per die a pool of expert slots (placement die i = opt.moe_dies[i]).
  // Each layer, a dispatch kernel splits the chosen experts into one list per die and one for
  // the CPU. No die waits for the host: the home die runs its list on a second stream next to
  // the shared expert; every other die gets the staged block (x + lists) through pinned host
  // memory as soon as it is there, reads its count on the device (0: exits early) and sends its
  // result back the same way, ordered by events. The host only runs the CPU's misses. The parts
  // are summed in a fixed order (an idle die contributes exact zeros).
  struct Die {
    int dev = -1;
    ExpertPool pool;
    void* scratch = nullptr;
    float* y_home = nullptr;     // its result on the home die
    // another die only:
    hipStream_t st = nullptr;
    hipEvent_t done = nullptr;
    char* d_in = nullptr;        // the staged block on dev
    float *d_y = nullptr, *h_y = nullptr;
  };
  std::vector<Die> dies;
  static constexpr int kLoadTestSlots = 8;
  std::unique_ptr<ExpertLoader> loader;  // the load test's
  std::vector<int> h_table;              // host copy of d_table (the cache edits it)
  std::unique_ptr<ExpertCache> cache;
  Placement placement;
  int* d_table = nullptr;       // [n_layer][n_expert]: die << 16 | slot, or -1 (home die)
  hipEvent_t ev_staged;         // the block (x, router outputs, lists) is in host memory
  hipStream_t s_local = nullptr;  // home die: its own expert list, next to the shared expert
  hipEvent_t ev_dispatched = nullptr, ev_local = nullptr;
  void setup_experts();

  // device scratch (one token)
  float *h, *h2, *pre, *a_pre, *a_post, *a_comb, *f_pre, *f_post, *f_comb, *mixes;
  float *x, *t1, *qr, *q, *kvv, *o, *oa, *y, *ysh, *yr, *logits_d, *partial;
  // activation codes + scales for the GEMVs, two sets: A holds a layer input (x) that several
  // GEMVs read, B the intermediate ones (qr, o, oa, act, engram rows); up to 32768 values each
  struct Codes { uint8_t* q; float* s; };
  Codes A, B;
  float *cmp_kvn, *cmp_sc, *lat, *ik, *iq, *iw, *iscore, *gate, *up, *act, *router_logits, *rw, *erows, *ekv;
  int *rids, *iota;
  std::map<int, int*> topk_buf;  // [index source] its selection buffers, [kChunk][idx_topk]
  std::vector<std::map<int, const int*>> topk_cur = std::vector<std::map<int, const int*>>(kChunk);
  // candidate blocks (the reference's two-level top-k, > 16k positions): per token of the chunk,
  // the candidate source's keep mask [max_ctx / block + 1], or null when every block is kept
  std::vector<const uint8_t*> cand_cur = std::vector<const uint8_t*>(kChunk, nullptr);
  uint8_t* cand_keep = nullptr;  // [kChunk][max_ctx / block + 1]
  float* cand_bscore = nullptr;  // [max_ctx / block + 1]
  int* cand_bidx = nullptr;      // [cand_topk_blocks]
                                 // [token of the chunk][index source] its selection (topk_buf or iota)
  // the routed experts' inputs, staged together so one copy moves them (device and pinned host):
  // lists [kChunk][kMoeListStride] (per token: the dies', then the CPU's), then x [kChunk][n_embd];
  // a chunk of M copies the lists and its M rows of x
  struct MoeIn { MoeList* lists; float* x; } d_moe, h_moe;
  size_t moe_lists_bytes = 0;
  size_t moe_block_bytes(int M) const { return moe_lists_bytes + (size_t)M * hp.n_embd * 4; }
  float *h_yr, *h_erows, *att_scratch;
  // canon: the expert output rows, compact per place ([place][kChunk * k][n_embd], place = die
  // index, n_dies = the CPU), and each token's rank -> (place, index) from the dispatch
  float* canon_rows = nullptr;
  int* canon_where = nullptr;

  FILE* route_log = nullptr;
  int* d_route = nullptr;     // [kChunk][n_layer][16] this forward's expert ids (device), then the weights
  int route_step = 0;         // forwards logged so far
  bool in_prefill = false;
  void write_route_records(const int32_t* toks, int M);

  // verification (verify / commit)
  bool verifying = false;
  bool chunked = false;  // in forward_chunk: per-token draft hiddens go to dspark->hid_chunk
  int verify_p0 = 0, verify_m = 0;
  float* ring_backup = nullptr;  // [n_layer][kMaxChunk][head_dim]: the ring slots a chunk overwrites
  // compressor partial groups (ratio > 1): a later rejected position can overwrite the slot of a
  // kept one (same pos % ratio), so each chunk position's row is kept, and the state before the
  // chunk; commit rebuilds every slot from the last kept position writing it
  std::map<int, float*> cmp_rows;     // [L] [kMaxChunk][kv | score][head_dim]
  std::map<int, float*> cmp_backup;   // [L] [kv | score][ratio][head_dim]

  int pos = 0;  // position of the current step
  int32_t token = 0;
  std::vector<int32_t> hist;  // tokens including the current one (engram)

  // M rows at once (a chunk of tokens): every row as a single-row call would give, bit for bit
  void quant(const float* xin, int n, Codes c, int M = 1);
  void gemv8(const Fp8W& w, Codes c, float* yout, int rows_per_group = 0, int M = 1);
  // the chunk's M tokens are at positions pos .. pos + M - 1; per-token buffers are [M][...]
  void attention(int L, const float* xin, float* out, int M = 1);
  void attn_core(int L, int t, const float* x_t, float* q_t, const float* kvv_t, Codes qr_t, float* o_t);
  // the routed + shared experts of the chunk's M tokens (x [M][E] staged in d_moe.x)
  void moe(int L, float* out, int M, Taps* taps);
  void moe_launch(int L, int M);                           // GPU: router .. the dies' experts, shared expert
  void moe_finish(int L, float* out, int M, Taps* taps);  // host: the CPU's experts, then the sum
  void block(int L, Taps* taps, int M = 1);
  void block_pre(int L, Taps* taps, int M);   // up to the routed experts started
  void block_post(int L, Taps* taps, int M);  // the CPU's experts onwards

  // A prompt chunk in two halves (CPU/GPU overlap, RtOptions::prefill_overlap): while the host
  // computes one half's CPU experts, the GPU runs the other half's attention. set_half() points
  // every per-token buffer at the half's first token (tok_ptrs), gives it its own residual
  // stream pair (h/h2 swap per sublayer) and its own events; tok_base offsets the per-token
  // indexer tables.
  struct TokPtr { void** p; size_t bytes; };  // a per-token buffer and its bytes per token
  std::vector<TokPtr> tok_ptrs;
  int tok_base = 0, cur_half = -1;
  float *half_h[2] = {}, *half_h2[2] = {};
  struct HalfEvents { hipEvent_t staged, dispatched, local; std::vector<hipEvent_t> done; } half_ev[2];
  void shift_tokens(int t0);
  void set_half(int i, int t0);  // i = -1: back to whole-chunk addressing
  void engram(int L, int M = 1);
  void fetch_engram_rows(const std::vector<int32_t>& chunk);
  std::vector<int32_t> forward_chunk(const std::vector<int32_t>& toks, bool head = true);  // the greedy next token after each
  void tap(Taps* taps, const char* name, const float* d, std::vector<int64_t> ne);
  void mark(const char* group);
  std::vector<std::pair<const char*, hipEvent_t>> marks;
  std::vector<hipEvent_t> ev_pool;
};
