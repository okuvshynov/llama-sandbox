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
  // routed experts on GPUs (expert_pool.h): a ranked placement file fills these dies' free
  // memory, keeping `reserve_gb` free on each; empty = every routed expert on the CPU
  std::string placement;
  std::vector<int> moe_dies = {0, 1, 2, 3};
  double reserve_gb = 0.75;
  // the DSpark draft head's GGUF (dspark.h); empty = none
  std::string dspark;
};

class Runtime {
 public:
  // fp8_path is fp8pack's output for the same model
  Runtime(const Model& m, RtOptions opt, const std::string& fp8_path);
  ~Runtime();

  // one step for `token` at position n_past(): returns the logits (empty when
  // want_logits is false: prompt tokens before the last skip the 1.3 GB head)
  std::vector<float> decode(int32_t token, bool want_logits = true);
  // the prompt, one token at a time (decode() for each); the last token's logits
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
  // (prompt tokens included): its position, id, whether it is a prompt token, and per layer the
  // router's expert ids and weights. Format in rt.cpp (Runtime::log_routing).
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
  std::map<int, int*> topk_buf;        // [index source] its selection buffer
  std::map<int, const int*> topk_cur;  // [index source] this step's selection (topk_buf or iota)
  // the routed experts' inputs, staged together so one copy moves them (device and pinned host):
  // x [n_embd], router ids [8], weights [8], lists [n_dies + 1] (the last: the CPU's)
  struct MoeIn { float* x; int* ids; float* w; MoeList* lists; } d_moe, h_moe;
  size_t moe_block_bytes = 0;
  float *h_yr, *h_erows, *att_scratch;

  FILE* route_log = nullptr;
  int* d_route = nullptr;     // [n_layer][8] this step's expert ids (device), then the weights
  bool in_prefill = false;
  void write_route_record();

  int pos = 0;  // position of the current step
  int32_t token = 0;
  std::vector<int32_t> hist;  // tokens including the current one (engram)

  void quant(const float* xin, int n, Codes c);
  void gemv8(const Fp8W& w, Codes c, float* yout, int rows_per_group = 0);
  void attention(int L, const float* xin, float* out);
  void moe(int L, const float* xin, float* out, Taps* taps);
  void block(int L, Taps* taps);
  void engram(int L);
  void fetch_engram_rows();
  void tap(Taps* taps, const char* name, const float* d, std::vector<int64_t> ne);
  void mark(const char* group);
  std::vector<std::pair<const char*, hipEvent_t>> marks;
  std::vector<hipEvent_t> ev_pool;
};
