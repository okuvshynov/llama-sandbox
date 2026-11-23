// ds-completion: DeepSeek-V4.1-Flash text completion.
//
//   ds-completion -m <model-00001-of-000NN.gguf> --rt <dense-fp8.gguf> (-p "prompt" | -f prompt.txt | -i | --requests)
//                 [--chat | --think [--effort 75]] [-n 128] [-t 16] [--max-ctx 65536] [--record run.json [--record-label k=v,...]]
//                 [--gpu D] [--placement file [--moe-dies 0,1,2,3] [--cache-loads R [--cache-period N] [--cache-window N] [--cache-min N] [--cache-wait]]] [--canon] [--profile]
//
// The prompt first, then the reply: the prompt in chunks of --prefill-chunk, then one token per
// step or, with --dspark --spec, several (the same output). Greedy unless --temp is given. --chat wraps the
// prompt as one user turn in the model's chat format without thinking, --think with it (the
// strings encoding/encoding.py produces; --effort is its reasoning effort, 1-100, default 75);
// otherwise the prompt is raw text after BOS. -f reads the prompt from a file.
// Only the tokenizer comes from libllama (vocab-only load).
// --requests (ds-completion, in place of a prompt): prompts from stdin, each going on from the
// context the last one left or from a saved point where it can, else on an empty context; the
// replies' tokens as lines on stdout (docs/requests.md; README steps 152, 158, 160).
// More options (sampling, speculation, traces): QUICKSTART.md, "Options".
// --record FILE: the run as one JSON object (docs/record.md): a few of its arguments, the prompt's
// token count and hash, the GPU clock setting, the bytes read from storage per phase, the
// speculative loop's totals, the routed experts' uses, placement and cache totals, and the times
// the "# ..." lines on stderr report.
// Built with make HIP=1. --rt <dense-fp8.gguf> is required: every token goes through the own
// runtime (hip/rt.h: the dense part on GPU D in exact fp8, routed experts on the CPU or, with
// --placement, partly on GPUs). (Until README step 193 this file also built ds-ref, the model on
// the CPU alone as ggml graphs: the first implementation, removed in steps 192-194.)

#include "fatal.h"
#include "llama.h"
#include "options.h"
#include "record.h"
#include "sampler.h"

#include <sstream>
#include "think_view.h"

#include "rt.h"

#include <algorithm>
#include <csignal>
#include <execinfo.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <memory>
#include <string>

using Clock = std::chrono::steady_clock;

static std::vector<int32_t> tokenize(const llama_vocab* v, const std::string& text) {
  std::vector<llama_token> t(text.size() + 16);
  int n = llama_tokenize(v, text.c_str(), (int32_t)text.size(), t.data(), (int32_t)t.size(),
                         /*add_special*/ false, /*parse_special*/ true);
  if (n < 0) {
    t.resize(-n);
    n = llama_tokenize(v, text.c_str(), (int32_t)text.size(), t.data(), (int32_t)t.size(), false, true);
  }
  t.resize(n);
  return std::vector<int32_t>(t.begin(), t.end());
}

// special: the text of a special token too (false: it prints as nothing)
static std::string piece(const llama_vocab* v, int32_t tok, bool special = false) {
  char buf[256];
  int n = llama_token_to_piece(v, tok, buf, sizeof buf, 0, special);
  return n > 0 ? std::string(buf, n) : std::string();
}

// The text of tokens [a, b) of v in quotes, on one line (special tokens written out, line ends
// and quotes escaped): for the --requests note on where a prompt parts from the context.
[[maybe_unused]] static std::string quoted(const llama_vocab* vocab, const std::vector<int32_t>& v, size_t a, size_t b) {
  std::string out = "\"";
  for (size_t i = a; i < b && i < v.size(); i++)
    for (unsigned char c : piece(vocab, v[i], true)) {
      char hex[8];
      if (c == '\n') out += "\\n";
      else if (c == '\r') out += "\\r";
      else if (c == '\t') out += "\\t";
      else if (c == '"' || c == '\\') out += '\\', out += (char)c;
      else if (c < 0x20) snprintf(hex, sizeof hex, "\\x%02x", c), out += hex;
      else out += (char)c;
    }
  return out + "\"";
}

// A fatal signal (an abort from glibc's heap checks, a crash) prints, before the default action,
// where the process was (fatal.h's phase), the thread (its name: the loader's are load-die<N>),
// the expert loader's loads per die and the stack (README steps 93, 96: a rare abort at exit; no
// core is kept on this machine). Only write(), no allocation. Addresses resolve with
// addr2line -f -C -e bin-hip/ds-completion <+offset>.
static void fatal_put(const char* s) {
  ssize_t w = write(2, s, strlen(s));
  (void)w;
}
static void fatal_put_long(long v) {
  char b[24];
  int n = sizeof b;
  b[--n] = 0;
  const bool neg = v < 0;
  unsigned long u = neg ? -(unsigned long)v : (unsigned long)v;
  do b[--n] = (char)('0' + u % 10); while (u /= 10);
  if (neg) b[--n] = '-';
  fatal_put(b + n);
}
static void on_fatal(int sig) {
  char name[32] = "?";
  (void)prctl(PR_GET_NAME, name);
  const long tid = syscall(SYS_gettid);
  fatal_put("# fatal signal ");
  fatal_put_long(sig);
  fatal_put("; phase: ");
  fatal_put(g_fatal_phase.load(std::memory_order_relaxed));
  fatal_put("; thread: ");
  fatal_put(name);
  fatal_put(tid == (long)getpid() ? " (the main thread)" : " (not the main thread)");
  fatal_put("; loads started / finished per die:");
  for (int d = 0; d < kFatalDies; d++) {
    fatal_put(" ");
    fatal_put_long(g_fatal_loads_started[d].load(std::memory_order_relaxed));
    fatal_put("/");
    fatal_put_long(g_fatal_loads_done[d].load(std::memory_order_relaxed));
  }
  fatal_put("\n# backtrace:\n");
  void* bt[64];
  backtrace_symbols_fd(bt, backtrace(bt, 64), 2);
  signal(sig, SIG_DFL);
  raise(sig);
}

// main's first local: its destructor runs after every other local's (the model's unmap among them)
struct AfterMain {
  ~AfterMain() { fatal_phase("after main: static destructors and the libraries' exit"); }
};

// The command line, as given (parse_args); n_predict is later capped by --max-ctx.
struct Args {
  std::string model_path, prompt, mode = "raw", rt_path;
  int n_predict = 128;
  int effort = 75;  // --think's reasoning effort (encoding.py's default)
  SamplerOptions sampler;  // --temp, --top-p, --seed (sampler.h); the default is greedy
  ThinkView::Mode think_show = ThinkView::All;  // --think-show: where the reasoning is printed (think_view.h)
  Options opt;
  int gpu = 0;
  bool profile = false;
  std::string trace_path;
  std::string route_log;
  std::string placement;
  std::vector<int> moe_dies = {0, 1, 2, 3};
  bool canon = false;
  bool max_ctx_set = false;
  int prefill_chunk = 512;
  bool prefill_overlap = true;
  double cache_loads = 0;
  int cache_period = 16, cache_window = 256, cache_min = 2;
  bool cache_wait = false;
  std::string dspark_path;
  bool spec = false;
  bool interactive = false;
  bool requests = false;  // --requests: prompts from stdin (docs/requests.md; README steps 152, 158, 160)
  int spec_n = 0;
  std::string spec_log_path;
  std::string record_path;
  std::vector<std::pair<std::string, std::string>> record_label;
};

// -1: go on; else the exit code (2: a usage error, 1: a prompt file that cannot be read)
static int parse_args(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&] { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
    if (a == "-m") args.model_path = next();
    else if (a == "-p") args.prompt = next();
    else if (a == "-f") {  // the prompt from a text file, as is (trailing newline included)
      const char* path = next();
      FILE* f = fopen(path, "rb");
      if (!f) { fprintf(stderr, "cannot read %s\n", path); return 1; }
      args.prompt.clear();
      char buf[65536];
      for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;) args.prompt.append(buf, n);
      fclose(f);
    }
    else if (a == "--effort") {  // --think: 1-100, or low / high / max (encoding.py: 50 / 75 / 100)
      const std::string e = next();
      args.effort = e == "low" ? 50 : e == "high" ? 75 : e == "max" ? 100 : atoi(e.c_str());
      if (args.effort < 1 || args.effort > 100) { fprintf(stderr, "--effort: 1-100, low, high or max\n"); return 2; }
    }
    else if (a == "--think-show") {  // all (default) / line / none
      if (!ThinkView::parse(next(), args.think_show)) { fprintf(stderr, "--think-show: all, line or none\n"); return 2; }
    }
    else if (a == "--chat") args.mode = "chat";
    else if (a == "--think") args.mode = "think";
    else if (a == "--requests") args.requests = true;  // runtime: requests on stdin, replies on stdout (docs/requests.md)
    else if (a == "-i" || a == "--interactive") args.interactive = true;  // runtime, --chat or --think: a session of turns on one context
    else if (a == "-n") args.n_predict = atoi(next());
    else if (a == "-t") args.opt.n_threads = atoi(next());
    else if (a == "--gpu") args.gpu = atoi(next());
    else if (a == "--dspark") args.dspark_path = next();  // the DSpark draft GGUF (alone: shadow mode, drafts measured, not used)
    else if (a == "--spec") args.spec = true;  // with --dspark: speculative decoding (the same output)
    else if (a == "--temp") args.sampler.temp = atof(next());    // 0 (default): greedy
    else if (a == "--top-p") args.sampler.top_p = atof(next());  // with --temp: the most probable tokens up to this probability
    else if (a == "--seed") args.sampler.seed = strtoull(next(), nullptr, 10);
    else if (a == "--spec-log") args.spec_log_path = next();  // per verify step: n k verify_ms draft_ms conf[0..]
    else if (a == "--spec-n") args.spec_n = atoi(next());  // fixed drafts verified per step (default: chosen per step)
    else if (a == "--placement") args.placement = next();  // runtime: routed experts on GPUs (expert_pool.h)
    else if (a == "--canon") args.canon = true;  // runtime: placement-invariant expert numerics (rt.h)
    else if (a == "--cache-loads") args.cache_loads = atof(next());  // runtime: adaptive expert cache (expert_cache.h)
    else if (a == "--cache-period") args.cache_period = atoi(next());
    else if (a == "--cache-window") args.cache_window = atoi(next());
    else if (a == "--cache-min") args.cache_min = atoi(next());  // runtime: a newcomer's least uses in the window
    else if (a == "--cache-wait") args.cache_wait = true;
    else if (a == "--moe-dies") {
      args.moe_dies.clear();
      for (char* q = (char*)next(); *q;) { args.moe_dies.push_back((int)strtol(q, &q, 10)); if (*q == ',') q++; }
    }
    else if (a == "--profile") args.profile = true;
    else if (a == "--trace") args.trace_path = next();  // runtime: a Chrome trace of every step (ui.perfetto.dev)
    else if (a == "--rt") args.rt_path = next();
    else if (a == "--log-routing") args.route_log = next();  // runtime: every token's experts per layer
    else if (a == "--max-ctx") args.max_ctx_set = true, args.opt.max_ctx = atoi(next());
    else if (a == "--prefill-chunk") args.prefill_chunk = atoi(next());  // runtime: prompt tokens per forward
    else if (a == "--no-overlap") args.prefill_overlap = false;  // runtime: prompt chunks without the CPU/GPU overlap
    else if (a == "--record") args.record_path = next();  // the run as one JSON object (docs/record.md)
    else if (a == "--record-label") {  // key=value[,key=value...]: the caller's names for the run, written back
      const std::string l = next();
      if (!label_add(l, args.record_label)) { fprintf(stderr, "--record-label: key=value[,key=value...], every key once; got %s\n", l.c_str()); return 2; }
    }
    else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  if (args.sampler.temp < 0 || !(args.sampler.top_p > 0 && args.sampler.top_p <= 1)) { fprintf(stderr, "--temp: 0 or more; --top-p: above 0, up to 1\n"); return 2; }
  if (args.sampler.temp == 0 && args.sampler.top_p < 1) { fprintf(stderr, "--top-p needs --temp (without it the choice is greedy)\n"); return 2; }
  if (args.sampler.temp > 0)
    fprintf(stderr, "# sampling: temperature %g, top-p %g, seed %llu\n", args.sampler.temp, args.sampler.top_p, (unsigned long long)args.sampler.seed);
  if (args.rt_path.empty() && !args.model_path.empty()) {
    fprintf(stderr, "ds-completion needs --rt <dense-fp8.gguf> (QUICKSTART.md)\n");
    return 2;
  }
  if (args.interactive && (args.mode == "raw" || args.rt_path.empty())) {
    fprintf(stderr, "-i needs --chat or --think, and --rt\n");
    return 2;
  }
  if (args.requests && (args.rt_path.empty() || args.mode != "raw" || !args.prompt.empty() || args.interactive ||
                        !args.record_path.empty())) {
    fprintf(stderr, "--requests needs --rt; not with -p, -f, -i, --chat, --think or --record\n");
    return 2;
  }
  if (args.model_path.empty() || (args.prompt.empty() && !args.interactive && !args.requests)) {
    fprintf(stderr, "usage: ds-completion -m <model.gguf> --rt <dense-fp8.gguf> (-p \"prompt\" | -f prompt.txt | -i | --requests) [--chat|--think [--effort 1-100|low|high|max] [--think-show all|line|none]] [-n N] [--temp T [--top-p P] [--seed S]] [-t N] [--max-ctx N] [--gpu D] [--placement file [--moe-dies 0,1,2,3] [--cache-loads R [--cache-period N] [--cache-window N] [--cache-min N] [--cache-wait]]] [--canon] [--profile]\n"
                    "more (--dspark/--spec, --trace, --record, ...): QUICKSTART.md, Options\n");
    return 2;
  }
  return -1;
}

// What --record writes of a turn beyond the arguments (docs/record.md).
struct RecordRun {
  const std::vector<int32_t>& toks;      // the prompt
  const std::vector<std::string>& sclk;  // the GPU clock setting as the run started
  const long long* rd;                   // [4] storage reads so far: at the start, after the load, the prompt, the generation
  double t_load, t_pp, t_tg;             // seconds: the load, the prompt, the generation
  int n_gen;
  const std::vector<int32_t>& gen;       // the generated ids (the first n_gen were printed)
  long spec_steps, spec_plain, spec_accepted;
  const std::vector<long>& spec_hist;    // verify steps by drafts verified
  double draft_ms, verify_ms;
};

// The run as one JSON object (docs/record.md, schema 1), and the file closed.
// false: the file could not be written.
static bool write_record(FILE* record, const Args& args, const RecordRun& r, const Runtime* rt) {
  fprintf(record, "{\n  \"schema\": 1,\n");
  fprintf(record, "  \"label\": %s,\n", label_json(args.record_label).c_str());
  fprintf(record, "  \"args\": {\"mode\": %s, \"effort\": %d, \"n\": %d, \"threads\": %d, \"runtime\": %s, \"canon\": %s, \"spec\": %s,\n"
                  "           \"placement\": %s, \"cache_loads\": %g, \"cache_period\": %d, \"cache_window\": %d, \"cache_min\": %d,\n"
                  "           \"cache_wait\": %s, \"spec_n\": %d, \"temp\": %g, \"top_p\": %g, \"seed\": %llu,\n"
                  "           \"profile\": %s, \"trace\": %s, \"spec_log\": %s, \"log_routing\": %s},\n",
          json_string(args.mode).c_str(), args.effort, args.n_predict, args.opt.n_threads, args.rt_path.empty() ? "false" : "true", args.canon ? "true" : "false",
          args.spec ? "true" : "false", json_string(args.placement).c_str(), args.cache_loads, args.cache_period, args.cache_window, args.cache_min,
          args.cache_wait ? "true" : "false", args.spec_n, args.sampler.temp, args.sampler.top_p, (unsigned long long)args.sampler.seed,  // as they took effect: an option not given has its default
          // the options that add work to a run (a record with one of them on is not a clean timing): on or off
          args.profile ? "true" : "false", args.trace_path.empty() ? "false" : "true", args.spec_log_path.empty() ? "false" : "true",
          args.route_log.empty() ? "false" : "true");
  fprintf(record, "  \"prompt\": {\"tokens\": %zu, \"hash\": \"fnv1a64:%016llx\"},\n", r.toks.size(),
          (unsigned long long)fnv1a64_ids(r.toks));
  fprintf(record, "  \"env\": {\"gpu_sclk\": %s, \"ds\": %s},\n", json_array(r.sclk).c_str(), env_ds_json().c_str());
  auto read = [&](int i) { return r.rd[i] < 0 || r.rd[i + 1] < 0 ? std::string("null") : std::to_string(r.rd[i + 1] - r.rd[i]); };
  fprintf(record, "  \"read_bytes\": {\"load\": %s, \"prompt\": %s, \"gen\": %s},\n", read(0).c_str(), read(1).c_str(), read(2).c_str());
  if (args.spec) {  // totals of the speculative loop; "verified": steps by the number of drafts verified, from 0
    std::vector<std::string> h;
    for (long c : r.spec_hist) h.push_back(std::to_string(c));
    fprintf(record, "  \"spec\": {\"steps\": %ld, \"plain_steps\": %ld, \"accepted\": %ld, \"verified\": %s, \"draft_ms\": %.1f, \"verify_ms\": %.1f},\n",
            r.spec_steps, r.spec_plain, r.spec_accepted, json_array(h).c_str(), r.draft_ms, r.verify_ms);
  } else {
    fprintf(record, "  \"spec\": null,\n");
  }
  {  // routed experts: uses by where they ran (prompt, generation and rejected drafts), the placement, the cache
    const Runtime::ExpertStats es = rt->expert_stats();
    std::vector<std::string> dv, pv;
    for (int d : es.dies) dv.push_back(std::to_string(d));
    for (int n : es.placed) pv.push_back(std::to_string(n));
    fprintf(record, "  \"experts\": {\"gpu_uses\": %ld, \"cpu_uses\": %ld, \"dies\": %s, \"placed\": %s,\n               \"cache\": ",
            rt->total_gpu_experts, rt->total_cpu_experts, json_array(dv).c_str(), json_array(pv).c_str());
    if (es.has_cache) {  // loads_started_by_die: in the order of "dies"; the home die takes none (README step 86)
      std::vector<std::string> lv;
      for (long n : es.cache.started_by_die) lv.push_back(std::to_string(n));
      fprintf(record, "{\"rebalances\": %ld, \"swaps_planned\": %ld, \"loads_started\": %ld, \"loads_published\": %ld,\n"
                      "                         \"loads_started_by_die\": %s}},\n",
              es.cache.rebalances, es.cache.swaps_planned, es.cache.loads_started, es.cache.loads_published, json_array(lv).c_str());
    }
    else
      fprintf(record, "null},\n");
  }
  {  // the generation as values: the ids printed, in order (not the end-of-sequence id that stopped it)
    const std::vector<int32_t> printed(r.gen.begin(), r.gen.begin() + std::min((size_t)r.n_gen, r.gen.size()));
    fprintf(record, "  \"output\": {\"tokens\": %zu, \"hash\": \"fnv1a64:%016llx\"},\n", printed.size(),
            (unsigned long long)fnv1a64_ids(printed));
  }
  fprintf(record, "  \"counters\": {\"load_s\": %.3f, \"prompt_s\": %.4f, \"prompt_tps\": %.3f, \"gen_tokens\": %d, \"gen_s\": %.4f, \"gen_tps\": %.3f}\n}\n",
          r.t_load, r.t_pp, r.toks.size() / r.t_pp, r.n_gen, r.t_tg, r.n_gen > 0 ? r.n_gen / r.t_tg : 0.0);
  return fclose(record) == 0;
}

// What one turn's generation did (a run without -i is one turn).
// --requests: SIGUSR1 stops the request being generated (docs/requests.md, README step 154).
// Set by the handler, read by the two generation loops between steps, cleared as a request is read.
static volatile sig_atomic_t g_cancel = 0;

struct Turn {
  int n_gen = 0;             // tokens printed
  bool hit_eos = false;
  bool cancelled = false;    // stopped by g_cancel
  std::vector<int32_t> gen;  // the generated ids (the plain loop also keeps the end-of-sequence id that ended it)
  // shadow mode (--dspark without --spec): after every generated token, the draft's guesses for the
  // tokens after it; compared with what the main model then generates (nothing is skipped or verified)
  std::vector<std::vector<int32_t>> drafts;
  std::vector<std::vector<float>> confs;
  double draft_ms = 0;
  // --spec: verify steps, drafts accepted, the time in verify steps, the steps by drafts verified,
  // the steps without a draft
  long spec_steps = 0, spec_accepted = 0;
  double verify_ms = 0;
  std::vector<long> spec_hist;
  long spec_plain = 0;
  ThinkView* view = nullptr;  // prints the reply
};

// --spec: generation by speculative decoding. logits: the prompt's last; first: a session's first turn
// (its logs are opened anew); pending: what was printed but not yet run through the model.
static void generate_spec(Runtime* rt, const Args& args, const llama_vocab* vocab, int eos, const std::vector<float>& logits, bool first,
                          std::vector<int32_t>& pending, Turn& turn) {
  // speculative decoding: draft from the anchor (the last emitted token, not yet run),
  // verify [anchor, drafts], keep the matching prefix plus the main model's own next token;
  // prints exactly what the plain loop prints. With --temp the main model's token at a position
  // is the sampler's (keyed by the position, so the same as in the plain loop), from the
  // chunk's logits on the host; a draft is accepted if it is that token (README step 150).
  const bool sampling = args.sampler.temp > 0;
  rt->keep_verify_logits = sampling;
  SamplerScratch scratch;
  int32_t anchor = sample_token(logits.data(), (int)logits.size(), args.sampler, rt->n_past());
  auto emit = [&](int32_t t) {  // false: stop
    if (t == eos) turn.hit_eos = true;
    if (t == eos || turn.n_gen >= args.n_predict) return false;
    turn.view->token(t, piece(vocab, t, args.requests));
    turn.gen.push_back(t);
    turn.n_gen++;
    return turn.n_gen < args.n_predict;
  };
  const char* log_mode = first ? "w" : "a";  // a session's later turns add to the logs
  FILE* spec_log = args.spec_log_path.empty() ? nullptr : fopen(args.spec_log_path.c_str(), log_mode);
  // README step 92: per verify step the draft's token ids and the ids it emitted (a study of the
  // draft's head over part of the vocabulary: results/decode-next/draft_vocab_sim.py)
  FILE* tok_log = getenv("DS_DRAFT_TOKEN_LOG") ? fopen(getenv("DS_DRAFT_TOKEN_LOG"), log_mode) : nullptr;
  if (tok_log) fprintf(tok_log, "# n_verified accepted verify_ms draft_ms | drafted ids (all of the block; none on a plain step) | emitted ids\n");
  if (spec_log) fprintf(spec_log, "# n_verified accepted verify_ms draft_ms conf[0..block)\n");
  // Draft length per step (unless --spec-n): the confidence head's sigmoid is the chance a draft
  // is accepted given the earlier ones were (calibrated: results/spec-conf/spec_conf.py, README step 25);
  // verify the n in 0..5 with the most expected tokens per ms, (1 + p0 + p0 p1 + ...) /
  // (draft + verify(n)); n = 0 is a plain step. Verify cost: a line a + b n, fitted by weighted
  // least squares to this run's steps (weights decay by 0.97 per step, ~30 steps) plus a weak
  // fixed prior, 52 + 28 n ms (sclk 2, study30) at n = 0..5 with a total weight of 3 steps.
  // A line, not one average per n: every step then updates the cost of every n, including
  // those the rule has not tried lately (with a good expert cache the per-token cost is lower).
  const int n_max = rt->draft_block();
  struct Fit { double w = 0, n = 0, t = 0, nn = 0, nt = 0; } obs, prior;
  for (int n = 0; n <= n_max; n++) {
    const double w = 3.0 / (n_max + 1), t = 52.0 + 28.0 * n;
    prior.w += w; prior.n += w * n; prior.t += w * t; prior.nn += w * n * n; prior.nt += w * n * t;
  }
  std::vector<double> verify_cost(n_max + 1);
  auto refit = [&] {
    const Fit f{obs.w + prior.w, obs.n + prior.n, obs.t + prior.t, obs.nn + prior.nn, obs.nt + prior.nt};
    const double b = (f.w * f.nt - f.n * f.t) / (f.w * f.nn - f.n * f.n), a = (f.t - b * f.n) / f.w;
    for (int n = 0; n <= n_max; n++) verify_cost[n] = a + b * n;
  };
  refit();
  double draft_cost = 12.0;
  std::vector<long> n_hist(verify_cost.size(), 0);
  auto choose_n = [&](const std::vector<float>& conf) {
    int best_n = 0;
    double best = -1, run = 1, expect = 1;
    for (int n = 0; n < (int)verify_cost.size(); n++) {
      if (n > 0) {
        run *= 1.0 / (1.0 + std::exp(-(double)conf[n - 1]));
        expect += run;
      }
      const double rate = expect / (draft_cost + verify_cost[n]);
      if (rate > best) { best = rate; best_n = n; }
    }
    return best_n;
  };
  // Backoff: after a step that chose n = 0, the next 1, 2, 4, .. (at most 8) steps are plain,
  // without drafting at all (the 12 ms draft would likely be wasted again); any step that
  // verifies a draft resets it. The draft's window is fed by every committed position either
  // way.
  int backoff = 0, skip = 0;
  long plain_steps = 0;
  if (g_cancel) {
    turn.cancelled = true;
  } else if (emit(anchor)) {
    Clock::time_point t_emit, td_start;
    bool have_t_emit = false;
    for (;;) {
      if (g_cancel) {  // between steps: everything printed is committed but the anchor
        turn.cancelled = true;
        break;
      }
      DraftResult d;
      double dms = 0;
      int n_verify = 0;
      td_start = Clock::now();
      if (skip > 0) {
        skip--;
        plain_steps++;
        d.ids = {anchor};
      } else {
        auto td = Clock::now();
        d = rt->draft(anchor);
        dms = std::chrono::duration<double, std::milli>(Clock::now() - td).count();
        turn.draft_ms += dms;
        draft_cost += 0.1 * (dms - draft_cost);
        n_verify = args.spec_n > 0 ? args.spec_n : choose_n(d.conf);
        if (args.spec_n == 0) {
          backoff = n_verify == 0 ? std::min(backoff == 0 ? 1 : 2 * backoff, 8) : 0;
          skip = backoff;
        }
      }
      std::vector<int32_t> chunk(d.ids.begin(), d.ids.end());  // [anchor, drafts]
      if ((int)chunk.size() > n_verify + 1) chunk.resize(n_verify + 1);  // verify only the first n drafts
      const int room = args.opt.max_ctx - rt->n_past();
      if ((int)chunk.size() > room) chunk.resize(std::max(room, 1));
      auto tv = Clock::now();
      if (have_t_emit) rt->trace_host("after commit: emit the tokens", t_emit, td_start, 3);  // README step 86
      const uint64_t p0 = rt->n_past();  // the anchor's position; the token after chunk[i] is at p0 + i + 1
      std::vector<int32_t> next = rt->verify(chunk);
      if (sampling) {  // (counted in the verify's time: the draft-length rule's cost per token has it)
        const auto ts = Clock::now();
        // one after another, up to the first draft that is not the sampled token: the rows after
        // it are not used (README step 151: cheaper than six threads woken for six draws)
        for (size_t i = 0; i < chunk.size(); i++) {
          next[i] = sample_token(rt->verify_logits(i), (int)logits.size(), args.sampler, p0 + i + 1, &scratch);
          if (i + 1 < chunk.size() && next[i] != chunk[i + 1]) break;
        }
        rt->trace_host("after verify: sample", ts, Clock::now(), 3);
      }
      const auto tv1 = Clock::now();
      const double vms = std::chrono::duration<double, std::milli>(tv1 - tv).count();
      turn.verify_ms += vms;
      int k = 0;
      while (k + 1 < (int)chunk.size() && chunk[k + 1] == next[k]) k++;
      {
        const double nv = chunk.size() - 1;
        obs = {0.97 * obs.w + 1, 0.97 * obs.n + nv, 0.97 * obs.t + vms, 0.97 * obs.nn + nv * nv, 0.97 * obs.nt + nv * vms};
        refit();
      }
      n_hist[chunk.size() - 1]++;
      if (args.profile) {
        fprintf(stderr, "\n# verify %d tokens: %.1f ms (cpu experts %.1f; experts gpu %d cpu %d); GPU timeline:", (int)chunk.size(),
                rt->last.total_ms, rt->last.cpu_moe_ms, rt->last.gpu_experts, rt->last.cpu_experts);
        for (auto& [g, ms] : rt->last_groups) fprintf(stderr, " %s %.1f", g.c_str(), ms);
      }
      if (spec_log && !d.conf.empty()) {
        fprintf(spec_log, "%d %d %.2f %.2f", (int)chunk.size() - 1, k, vms, dms);
        for (float c : d.conf) fprintf(spec_log, " %.4f", c);
        fprintf(spec_log, "\n");
      }
      if (tok_log) {
        fprintf(tok_log, "%d %d %.2f %.2f |", (int)chunk.size() - 1, k, vms, dms);
        for (size_t i = 1; i < d.ids.size(); i++) fprintf(tok_log, " %d", d.ids[i]);
        fprintf(tok_log, " |");
        for (int i = 0; i <= k; i++) fprintf(tok_log, " %d", next[i]);
        fprintf(tok_log, "\n");
      }
      rt->trace_host("after verify: draft rule, logs", tv1, Clock::now(), 3);
      // the context keeps exactly what is printed: accepted drafts only up to an end of sequence
      // among them and up to -n's limit (the output is the same; an -i session continues from it)
      int keep = std::min(k, args.n_predict - turn.n_gen - 1);
      for (int i = 0; i < keep; i++)
        if (next[i] == eos) keep = i;
      rt->commit(keep + 1);
      t_emit = Clock::now();
      have_t_emit = true;
      turn.spec_steps++;
      turn.spec_accepted += k;
      bool go = true;
      for (int i = 0; i <= keep && go; i++) go = emit(next[i]);
      if (!go) break;
      anchor = next[keep];
    }
  }
  if (!turn.hit_eos && !turn.gen.empty()) pending = {turn.gen.back()};  // stopped by -n: printed, not yet run
  if (spec_log) fclose(spec_log);
  if (tok_log) fclose(tok_log);
  turn.spec_hist = n_hist;
  turn.spec_plain = plain_steps;
  fprintf(stderr, "\n# drafts verified per step:");
  for (size_t n = 0; n < n_hist.size(); n++) fprintf(stderr, " %zu: %ld", n, n_hist[n]);
  fprintf(stderr, " (of the 0s, %ld plain steps without a draft); verify cost fit %.1f + %.1f n ms", plain_steps,
          verify_cost[0], verify_cost[1] - verify_cost[0]);
}

// Plain generation, a token per step, greedy or sampled (--temp): on the runtime (with a draft
// loaded, in shadow mode). pos: the context's length as the loop starts,
// which is the first generated token's position (the sampler's random number is keyed by it).
static void generate_plain(Runtime* rt, const Args& args, const llama_vocab* vocab, int eos, std::vector<float>& logits,
                           uint64_t pos, std::vector<int32_t>& pending, Turn& turn) {
  for (; turn.n_gen < args.n_predict; turn.n_gen++, pos++) {
    if (g_cancel) {
      turn.cancelled = true;
      break;
    }
    const int32_t next = sample_token(logits.data(), (int)logits.size(), args.sampler, pos);
    turn.gen.push_back(next);
    if (next == eos) break;
    turn.view->token(next, piece(vocab, next, args.requests));
    if (turn.n_gen + 1 == args.n_predict) { turn.n_gen++; pending = {next}; break; }
    if (rt->has_draft()) {
      auto td = Clock::now();
      DraftResult d = rt->draft(next);
      turn.draft_ms += std::chrono::duration<double, std::milli>(Clock::now() - td).count();
      turn.drafts.emplace_back(d.ids.begin() + 1, d.ids.end());
      turn.confs.push_back(d.conf);
    }
    logits = rt->decode(next);
    if (args.profile) {
      fprintf(stderr, "\n# step %d: %.1f ms (cpu experts %.1f, waiting for the GPU %.1f; experts gpu %d cpu %d); GPU timeline:",
              turn.n_gen, rt->last.total_ms, rt->last.cpu_moe_ms, rt->last.wait_ms, rt->last.gpu_experts, rt->last.cpu_experts);
      for (auto& [g, ms] : rt->last_groups) fprintf(stderr, " %s %.1f", g.c_str(), ms);
    }
  }
}

// Shadow mode's result: how many of each draft's tokens the main model then generated.
static void report_shadow(const Turn& turn) {
  if (turn.drafts.empty()) return;
  // draft g (anchor = gen[g]) guesses gen[g + 1 ..]; its accepted length = the matching prefix
  const int B = (int)turn.drafts[0].size();
  std::vector<int> hist(B + 1, 0);
  long n = 0, acc_sum = 0;
  std::vector<long> pos_ok(B, 0), pos_n(B, 0);
  for (size_t g = 0; g < turn.drafts.size(); g++) {
    if (g + B >= turn.gen.size()) break;  // only drafts whose whole block can be checked
    int a = 0;
    while (a < B && turn.drafts[g][a] == turn.gen[g + 1 + a]) a++;
    hist[a]++;
    acc_sum += a;
    n++;
    for (int k = 0; k < B; k++) { pos_n[k]++; pos_ok[k] += turn.drafts[g][k] == turn.gen[g + 1 + k]; }
  }
  if (n > 0) {
    fprintf(stderr, "# dspark shadow: %ld drafts of %d; accepted prefix mean %.2f -> %.2f tokens per main step; %.1f ms per draft\n",
            n, B, (double)acc_sum / n, 1.0 + (double)acc_sum / n, turn.draft_ms / turn.drafts.size());
    fprintf(stderr, "#   accepted 0..%d:", B);
    for (int a = 0; a <= B; a++) fprintf(stderr, " %d", hist[a]);
    fprintf(stderr, "\n#   per position correct:");
    for (int k = 0; k < B; k++) fprintf(stderr, " %.0f%%", 100.0 * pos_ok[k] / pos_n[k]);
    fprintf(stderr, "\n");
  }
}

static int run(int argc, char** argv) {
  AfterMain after_main;
  {
    void* bt[1];
    (void)backtrace(bt, 1);  // its first call loads libgcc (it allocates): not inside the handler, after a corruption
  }
  for (int sig : {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE}) signal(sig, on_fatal);
  Args args;
  if (const int rc = parse_args(argc, argv, args); rc >= 0) return rc;
  // --record describes one prompt and its generation: not a session.
  // Opened now, written after the generation: a run that ends early leaves it empty.
  FILE* record = nullptr;
  if (args.record_path.empty() && !args.record_label.empty()) { fprintf(stderr, "--record-label needs --record\n"); return 2; }
  if (!args.record_path.empty()) {
    if (args.interactive) { fprintf(stderr, "--record: not with -i\n"); return 2; }
    record = fopen(args.record_path.c_str(), "w");
    if (!record) { fprintf(stderr, "cannot write %s\n", args.record_path.c_str()); return 1; }
  }
  const std::vector<std::string> sclk = record ? gpu_sclk() : std::vector<std::string>();  // as the run starts
  // storage reads per phase (record.h): the counter at the start, after the load, the prompt, the generation
  long long rd[4] = {record ? io_read_bytes() : -1, -1, -1, -1};

  // tokenizer only
  llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.vocab_only = true;
  llama_model* tok_model = llama_model_load_from_file(args.model_path.c_str(), mp);
  if (!tok_model) { fprintf(stderr, "cannot load the tokenizer from %s\n", args.model_path.c_str()); return 1; }
  const llama_vocab* vocab = llama_model_get_vocab(tok_model);

  const std::string bos = "<｜begin▁of▁sentence｜>";
  // a user turn as encoding.py renders it; the first one carries BOS and, with --think, the effort
  auto turn_text = [&](const std::string& user, bool first) {
    std::string t = first ? bos : "";
    if (first && args.mode == "think")
      t += "<｜System｜>Reasoning Effort: " + std::to_string(args.effort) +
           " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
    return t + "<｜User｜>" + user + "<｜Assistant｜>" + (args.mode == "think" ? "<think>" : "</think>");
  };
  const std::string text = args.mode == "raw" ? bos + args.prompt : turn_text(args.prompt, true);
  std::vector<int32_t> toks;
  if (!args.prompt.empty()) toks = tokenize(vocab, text);  // -i without a prompt: the first turn is read after the load
  // the GPU runtime defaults to the model's original training context (YaRN extends it 16x, to
  // 1M: --max-ctx); ~6.4 KB per position on the home die (compressed caches, index keys). The CPU
  // path keeps 4096 (it builds dense masks per prompt and stops at 16k: no candidate blocks)
  if (!args.rt_path.empty() && !args.max_ctx_set) args.opt.max_ctx = 65536;

  if ((int)toks.size() >= args.opt.max_ctx) {
    fprintf(stderr, "prompt is %zu tokens; --max-ctx is %d\n", toks.size(), args.opt.max_ctx);
    return 1;
  }
  if ((int)toks.size() + args.n_predict > args.opt.max_ctx) {
    args.n_predict = args.opt.max_ctx - (int)toks.size();
    fprintf(stderr, "# generation capped at %d tokens by --max-ctx %d\n", args.n_predict, args.opt.max_ctx);
  }
  Model m;
  auto t0 = Clock::now();
  try {
    m.load(args.model_path);
  } catch (const std::exception& e) {
    fprintf(stderr, "load failed: %s\n", e.what());
    return 1;
  }
  {  // the experts that may run on the CPU: page tables now, not inside decode steps
    auto tm = Clock::now();
    const size_t n = m.map_experts(args.opt.n_threads);
    fprintf(stderr, "# mapped %.0f GB of CPU experts in %.1f s\n", n / 1e9,
            std::chrono::duration<double>(Clock::now() - tm).count());
  }
  std::unique_ptr<Runtime> rt;
  if (!args.rt_path.empty()) {  // (always: parse_args requires it)
    RtOptions ro;
    ro.gpu = args.gpu;
    ro.n_threads = args.opt.n_threads;
    ro.max_ctx = args.opt.max_ctx;
    ro.placement = args.placement;
    ro.moe_dies = args.moe_dies;
    ro.canon = args.canon;
    ro.prefill_chunk = args.prefill_chunk;
    ro.prefill_overlap = args.prefill_overlap;
    ro.cache_loads = args.cache_loads;
    ro.cache_period = args.cache_period;
    ro.cache_window = args.cache_window;
    ro.cache_min = args.cache_min;
    ro.cache_wait = args.cache_wait;
    ro.dspark = args.dspark_path;
    if (args.spec && args.dspark_path.empty()) { fprintf(stderr, "--spec needs --dspark\n"); return 2; }
    rt.reset(new Runtime(m, ro, args.rt_path));
    rt->profile = args.profile;
    if (!args.trace_path.empty()) rt->set_trace(args.trace_path);
    if (!args.route_log.empty()) rt->log_routing(args.route_log);
  }
  const double t_load = std::chrono::duration<double>(Clock::now() - t0).count();
  if (record) rd[1] = io_read_bytes();
  fprintf(stderr, "# loaded in %.1f s; prompt %zu tokens; fake-quant on; %d threads\n", t_load, toks.size(), args.opt.n_threads);

  // One turn, or with -i a session: every turn's tokens are appended to the same context, as
  // encoding.py renders a chat: the reply's end of sequence, then <｜User｜>line<｜Assistant｜></think>.
  // So a turn's reply is what a fresh run on the whole conversation would print.
  // With --think the reference drops earlier turns' reasoning: a finished turn is rendered
  // <｜Assistant｜></think>answer. So each turn marks the context after its <｜Assistant｜>, and the
  // next one rewinds there and runs </think>, the answer's tokens, then its own (the answer runs
  // again: its states depended on the reasoning).
  const std::vector<int32_t> think_end_toks = tokenize(vocab, "</think>");
  if (think_end_toks.size() != 1) { fprintf(stderr, "</think> is not one token in this vocabulary\n"); return 1; }
  const int32_t think_end = think_end_toks[0];
  std::vector<int32_t> answer;  // --think: the last reply after its </think>
  ThinkView view;
  view.mode = args.think_show;
  view.think_end = think_end;
  const int eos = llama_vocab_eos(vocab);
  [[maybe_unused]] const int n_predict_turn = args.n_predict;
  std::vector<int32_t> pending;  // printed by the last turn but not yet run through the model (it stopped at -n)
  if (args.requests) {  // the empty context, to return to when a request cannot go on from what is kept
    signal(SIGUSR1, [](int) { g_cancel = 1; });
    rt->prefill_stop = [] { return g_cancel != 0; };
    rt->mark();
    view.mode = ThinkView::Ids;
    fprintf(stderr, "# ready for requests on stdin (docs/requests.md)\n");
    printf("ready\n");
    fflush(stdout);
  }
  [[maybe_unused]] long n_requests = 0;
  [[maybe_unused]] size_t kept = 0;  // --requests: the prompt's first tokens that the context already held
  // --requests (README step 160): the tokens before the runtime's second saved point (empty:
  // there is none), where this request puts it (0: nowhere), the fewest tokens worth a point
  [[maybe_unused]] std::vector<int32_t> saved;
  [[maybe_unused]] size_t save_at = 0;
  [[maybe_unused]] const size_t kMinSaved = 64;
  for ([[maybe_unused]] bool first = true;; first = false) {
    if (args.requests) {
      // "gen n=N [temp=T] [top_p=P] [seed=S] [keep=0|1] bytes=B" and then B bytes: the whole prompt as text,
      // special tokens written out, the first one included. The end of input ends the run.
      std::string head, err;
      for (int c; (c = getchar()) != EOF;) {
        if (c == '\n') {
          if (head.empty()) continue;  // empty lines between requests
          break;
        }
        head += (char)c;
      }
      if (head.empty()) break;
      long n = -1, bytes = -1, keep = 1;
      SamplerOptions so;
      {
        std::istringstream in(head);
        std::string w;
        in >> w;
        if (w != "gen") err = "unknown request (expected: gen n=N [temp=T] [top_p=P] [seed=S] [keep=0|1] bytes=B)";
        while (err.empty() && in >> w) {
          const size_t eq = w.find('=');
          const std::string k = w.substr(0, eq), v = eq == std::string::npos ? "" : w.substr(eq + 1);
          char* end = nullptr;
          if (k == "n") n = strtol(v.c_str(), &end, 10);
          else if (k == "bytes") bytes = strtol(v.c_str(), &end, 10);
          else if (k == "keep") keep = strtol(v.c_str(), &end, 10);
          else if (k == "temp") so.temp = strtof(v.c_str(), &end);
          else if (k == "top_p") so.top_p = strtof(v.c_str(), &end);
          else if (k == "seed") so.seed = strtoull(v.c_str(), &end, 10);
          if (!end || end == v.c_str() || *end) err = "bad field " + w;
        }
      }
      if (err.empty() && bytes < 0) err = "no bytes= in the request";
      if (!err.empty()) {
        // A header that was not understood (an unknown first word, a field that is unknown or not a
        // number, no length): where its prompt ends is not known, or not to be trusted, so the input
        // cannot be followed any further. (Until step 183 a bad field after bytes= was answered
        // without reading the prompt, which was then read as the next header.)
        printf("error %s; closing\n", err.c_str());
        fflush(stdout);
        break;
      }
      std::string text(bytes, '\0');
      if (fread(text.data(), 1, bytes, stdin) != (size_t)bytes) break;  // the input ended inside a prompt
      if (n < 1) err = "n: 1 or more";
      if (err.empty() && (so.temp < 0 || !(so.top_p > 0 && so.top_p <= 1))) err = "temp: 0 or more; top_p: above 0, up to 1";
      if (err.empty()) {
        toks = tokenize(vocab, text);
        if (toks.empty()) err = "an empty prompt";
        else if ((int)toks.size() >= args.opt.max_ctx) err = "the prompt is " + std::to_string(toks.size()) + " tokens; --max-ctx is " + std::to_string(args.opt.max_ctx);
      }
      if (!err.empty()) {  // a value out of range, or the prompt: its bytes were read, the next request follows
        printf("error %s\n", err.c_str());
        fflush(stdout);
        continue;
      }
      g_cancel = 0;  // a signal that came after the last request ended was for that one
      first = n_requests++ == 0;  // (the logs are opened by the first request that runs)
      // README step 158: the context still holds the last request's prompt and the tokens of its
      // reply that were run. A prompt that starts with all of them goes on from there: only its
      // other tokens run, and the reply is the same (the state at a position does not depend on
      // how the tokens before it were split into forwards). Any other prompt: the empty context.
      // README step 160: a prompt that does not go on from the context may still share its start
      // with it (another conversation with the same system prompt and tools). The runtime's
      // second saved point is put at the end of what two such prompts share, when the second of
      // them runs; a later prompt that starts with those tokens returns to it. It stays where it
      // is until a prompt comes that does not start with its tokens: that one runs from the
      // empty context over the positions before the point, which ends the point (rt.h, mark).
      // README step 164: the choice is reported, in the "prompt" line (ctx, common, how) and on
      // stderr, with the tokens around the place where the prompt parts from the context.
      save_at = 0;
      const char* how = "empty";
      size_t n_ctx = 0, with_ctx = 0;
      {
        const std::vector<int32_t>& ctx = rt->tokens();
        auto common = [&](const std::vector<int32_t>& a) {
          size_t i = 0;
          while (i < a.size() && i < toks.size() && a[i] == toks[i]) i++;
          return i;
        };
        n_ctx = ctx.size();
        with_ctx = common(ctx);
        std::string note;  // what becomes of the saved point
        if (!saved.empty() && common(saved) < saved.size()) {
          note = "; the saved point at " + std::to_string(saved.size()) + " ends (the prompt starts with " + std::to_string(common(saved)) + " of its tokens)";
          saved.clear();
        }
        const bool go_on = keep && !ctx.empty() && with_ctx == ctx.size() && ctx.size() < toks.size();
        const bool to_saved = !go_on && keep && !saved.empty() && saved.size() < toks.size();
        if (go_on) {
          how = "continue";
          kept = ctx.size();
        } else if (to_saved) {
          how = "saved";
          kept = saved.size();
        } else {
          kept = 0;
          if (keep && saved.empty() && with_ctx >= kMinSaved && with_ctx < toks.size()) save_at = with_ctx;
        }
        if (ctx.empty()) fprintf(stderr, "# context: empty; the prompt has %zu tokens: ", toks.size());
        else fprintf(stderr, "# context: held %zu tokens, the prompt (%zu tokens) starts with %s%zu of them: ", n_ctx, toks.size(), with_ctx == n_ctx ? "all " : "", with_ctx);
        if (go_on) fprintf(stderr, "it goes on from the context (%zu kept)", kept);
        else if (to_saved) fprintf(stderr, "it returns to the saved point (%zu kept)", kept);
        else fprintf(stderr, "it starts on the empty context%s%s", !keep ? " (keep=0)" : !ctx.empty() && with_ctx == toks.size() ? " (it has no token after the context's)" : "",
                     save_at ? ("; a point is saved at " + std::to_string(save_at)).c_str() : "");
        fprintf(stderr, "%s\n", note.c_str());
        const size_t kShow = 8;  // tokens of text on each side of the place
        if (with_ctx < n_ctx && with_ctx < toks.size())
          fprintf(stderr, "# context: they part at token %zu: the context has %d %s, the prompt %d %s; before it %s, then the context %s, the prompt %s\n", with_ctx,
                  ctx[with_ctx], quoted(vocab, ctx, with_ctx, with_ctx + 1).c_str(), toks[with_ctx], quoted(vocab, toks, with_ctx, with_ctx + 1).c_str(),
                  quoted(vocab, toks, with_ctx - std::min(with_ctx, kShow), with_ctx).c_str(), quoted(vocab, ctx, with_ctx, with_ctx + kShow).c_str(),
                  quoted(vocab, toks, with_ctx, with_ctx + kShow).c_str());
        else if (with_ctx < n_ctx)
          fprintf(stderr, "# context: the prompt ends inside the context, at token %zu; the context goes on %s\n", with_ctx, quoted(vocab, ctx, with_ctx, with_ctx + kShow).c_str());
        if (to_saved) rt->rewind(1);  // (after the lines above: ctx is the runtime's own)
        else if (!go_on) rt->rewind();
      }
      pending.clear();
      args.sampler = so;
      args.n_predict = (int)std::min<long>(n, args.opt.max_ctx - (long)toks.size());
      printf("prompt tokens=%zu n=%d kept=%zu ctx=%zu common=%zu how=%s\n", toks.size(), args.n_predict, kept, n_ctx, with_ctx, how);
      fflush(stdout);
    }
    if (args.interactive && (!first || toks.empty())) {
      fprintf(stderr, "\n> ");
      char* line = nullptr;
      size_t cap = 0;
      ssize_t len = getline(&line, &cap, stdin);
      while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) len--;
      const std::string user(line ? line : "", len > 0 ? len : 0);
      free(line);
      if (user.empty()) break;  // an empty line or the end of input ends the session
      if (args.mode == "think" && !first) {
        rt->rewind();
        toks = {think_end};
        toks.insert(toks.end(), answer.begin(), answer.end());
      } else {
        toks = pending;
      }
      pending.clear();
      if (!first) toks.push_back(eos);
      const std::vector<int32_t> turn = tokenize(vocab, turn_text(user, first));
      toks.insert(toks.end(), turn.begin(), turn.end());
      const int room = args.opt.max_ctx - rt->n_past() - (int)toks.size();
      if (room < 1) { fprintf(stderr, "# the context is full (--max-ctx %d)\n", args.opt.max_ctx); break; }
      args.n_predict = std::min(n_predict_turn, room);
    }
    auto t1 = Clock::now();
    std::vector<float> logits;
    if (args.interactive && args.mode == "think") {  // the mark: after <｜Assistant｜>, before <think>
      rt->prefill(std::vector<int32_t>(toks.begin(), toks.end() - 1));
      rt->mark();
      logits = rt->prefill({toks.back()});
    } else {
      if (save_at) {  // (--requests) up to the shared start, the saved point, then the rest
        if (!rt->prefill(std::vector<int32_t>(toks.begin(), toks.begin() + save_at)).empty()) {  // (empty: cancelled)
          rt->mark(1);
          saved.assign(toks.begin(), toks.begin() + save_at);
          logits = rt->prefill(std::vector<int32_t>(toks.begin() + save_at, toks.end()));
        }
      } else {
        logits = rt->prefill(kept ? std::vector<int32_t>(toks.begin() + kept, toks.end()) : toks);
      }
    }
    const double t_pp = std::chrono::duration<double>(Clock::now() - t1).count();
    if (record) rd[2] = io_read_bytes();
    auto t2 = Clock::now();
    Turn turn;
    turn.view = &view;
    view.begin(args.mode == "think");
    if (logits.empty()) turn.cancelled = true;  // --requests: the prompt was stopped between its chunks
    else if (args.spec) generate_spec(rt.get(), args, vocab, eos, logits, first, pending, turn);
    else generate_plain(rt.get(), args, vocab, eos, logits, rt->n_past(), pending, turn);
    const double t_tg = std::chrono::duration<double>(Clock::now() - t2).count();
    if (record) rd[3] = io_read_bytes();
    if (turn.spec_steps)
      fprintf(stderr, "\n# speculative: %ld verify steps, %.2f drafts accepted per step, %.2f tokens per step; draft %.1f ms per step\n",
              turn.spec_steps, (double)turn.spec_accepted / turn.spec_steps, (double)turn.n_gen / turn.spec_steps, turn.draft_ms / turn.spec_steps);
    view.end();  // (after the lines above: the loop's last stderr line is ended by them)
    if (args.requests) {
      const bool ended = turn.hit_eos || (!turn.gen.empty() && turn.gen.back() == eos && (int)turn.gen.size() > turn.n_gen);
      printf("end stop=%s prompt_tokens=%zu kept=%zu saved=%zu gen_tokens=%d prompt_s=%.4f gen_s=%.4f\n", turn.cancelled ? "cancel" : ended ? "eos" : "length",
             toks.size(), kept, saved.size(), turn.n_gen, t_pp, t_tg);
    } else {
      printf("\n");
    }
    fflush(stdout);
    report_shadow(turn);
    if (rt->total_gpu_experts + rt->total_cpu_experts > 0)
      fprintf(stderr, "# routed experts: %.1f%% on GPUs (%ld of %ld)\n",
              100.0 * rt->total_gpu_experts / (rt->total_gpu_experts + rt->total_cpu_experts), rt->total_gpu_experts,
              rt->total_gpu_experts + rt->total_cpu_experts);
    if (kept) fprintf(stderr, "# prompt: %zu of its %zu tokens were in the context\n", kept, toks.size());
    fprintf(stderr, "# prompt: %zu tokens in %.2f s = %.2f t/s; generation: %d tokens in %.2f s = %.2f t/s\n",
            toks.size() - kept, t_pp, (toks.size() - kept) / t_pp, turn.n_gen, t_tg, turn.n_gen > 0 ? turn.n_gen / t_tg : 0.0);
    if (record) {
      const RecordRun r{toks, sclk, rd, t_load, t_pp, t_tg, turn.n_gen, turn.gen, turn.spec_steps, turn.spec_plain, turn.spec_accepted, turn.spec_hist, turn.draft_ms, turn.verify_ms};
      const Runtime* rtp = rt.get();
      if (!write_record(record, args, r, rtp)) { fprintf(stderr, "cannot write %s\n", args.record_path.c_str()); return 1; }
      record = nullptr;
    }
    if (!args.interactive && !args.requests) break;
    {  // --think: what the next turn keeps of this reply
      auto it = std::find(turn.gen.begin(), turn.gen.end(), think_end);
      answer.assign(it == turn.gen.end() ? it : it + 1, turn.gen.end());
      if (!answer.empty() && answer.back() == eos) answer.pop_back();
    }
  }
  fatal_phase("main: freeing the tokenizer");
  llama_model_free(tok_model);
  llama_backend_free();
  rt.reset();  // here, not among main's locals, so the phases around it are exact
  fatal_phase("main: its locals (the model's unmap)");
  return 0;
}

// Errors after startup (a chunk past --max-ctx, a rewind without a mark)
// end the run with their message; unwinding closes the runtime, so a --trace is still written.
int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    fprintf(stderr, "\nerror: %s\n", e.what());
    return 1;
  }
}
