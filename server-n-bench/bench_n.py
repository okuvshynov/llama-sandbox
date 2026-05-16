#!/usr/bin/env python3
"""
Perf test for the server's OpenAI-style `n` (multiple completions per request).

Two modes:

  ./bench_n.py [max_tokens]                          A/B: parallel n=N vs N serial
  ./bench_n.py --sweep [max_tokens] [nlist] [reps]   scaling curve over n values

`reps` (default 1) runs each n value that many times; the sweep then reports the
median run and the min-max spread, so single-sample noise is visible.

Pass `--jsonl PATH` (any position before the positional args) to append every
individual run as one JSONL row to PATH -- so reps>1 preserves variance instead
of collapsing to the median. In sweep mode that's REPS rows per n value; in A/B
mode it's the parallel run plus one row per sequential request.

Pass `--n-prompt N` (any position) to build a prompt that tokenizes to ~N
tokens via /tokenize on a sibling seed corpus (default seed_corpus.cpp -- a
few canonical algorithms in C++; override with the BENCH_N_SEED env var),
repeated and truncated to exactly N tokens, then detokenized back to text.
Code is used rather than prose so that MoE expert routing matches what
would happen on a real coding workload. The chat template adds a few
wrapper tokens on the wire; the actual `prompt_n` ends up in each jsonl
row's timings so the on-the-wire count is recoverable. Without --n-prompt,
the original made-up technical-writer prompt is used.

"Cold" runs are guaranteed by erasing every slot's prompt cache via
POST /slots/{id}?action=erase right before each measured run, instead of by
salting the prompt with a per-run tag. One shared prompt is reused everywhere.
  - parallel  : erase all slots, then 1 request n=N -> prompt processed once,
                N generations batched
  - sequential: erase all slots once, then N requests -> prompt on req #1,
                reused for #2..N (no clear between them), serial gen
  - sweep     : erase all slots before each individual run

Server must be started with: -np <max n you want to test> --metrics --slot-save-path PATH

(--slot-save-path is required because it gates the entire /slots/* action
endpoint -- erase doesn't actually write to PATH, but the flag must be set.
server-context.cpp:3572)

NOTE on the /metrics prompt_tokens_total counter: for an n>1 request the server
copies the parent slot's prompt-token count into every child slot
(copy_state_to, server-context.cpp:562), so that counter reports ~n x the real
work. This script reads real prompt cost from each response's `timings` block
and only uses /metrics for decode-call counts (which are correct).
"""
import json, os.path, time, sys, urllib.request, urllib.error

BASE = "http://127.0.0.1:8080"
SEED = 1234

# ---- args -------------------------------------------------------------------
argv = sys.argv[1:]
JSONL_PATH = None
N_PROMPT   = None  # if set, build a prompt that tokenizes to ~N_PROMPT tokens
# pull --jsonl PATH and --n-prompt N out of argv from any position; everything
# else stays positional.
i = 0
while i < len(argv):
    if argv[i] == "--jsonl" and i + 1 < len(argv):
        JSONL_PATH = argv[i + 1]
        del argv[i:i + 2]
    elif argv[i] == "--n-prompt" and i + 1 < len(argv):
        N_PROMPT = int(argv[i + 1])
        del argv[i:i + 2]
    else:
        i += 1
SWEEP = argv and argv[0] == "--sweep"
if SWEEP:
    argv = argv[1:]
MAX_TOKENS = int(argv[0]) if len(argv) > 0 else 128
N          = int(argv[1]) if (not SWEEP and len(argv) > 1) else 16
NLIST      = [int(x) for x in argv[1].split(",")] if (SWEEP and len(argv) > 1) \
             else [1, 2, 4, 8, 16, 32, 64]
REPS       = int(argv[2]) if (SWEEP and len(argv) > 2) else 1

META = {}  # populated from /props once SLOTS is fetched (see bottom of helpers)

def jdump(row):
    if JSONL_PATH is None:
        return
    row = {"ts": time.time(), **META, **row}
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")

# ---- helpers ----------------------------------------------------------------
# Seed corpus for --n-prompt is read from a sibling file at startup. Default
# is seed_corpus.cpp (a few canonical algorithms in C++) -- representative of
# a coding workload, which matters for MoE models where expert routing depends
# on input content. For prose-heavy benchmarks, override via env var
# BENCH_N_SEED.
SEED_PATH = os.environ.get(
    "BENCH_N_SEED",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "seed_corpus.cpp"),
)

def make_default_prompt():
    body = ("The llama.cpp server schedules requests onto a fixed pool of slots. "
            "Each slot owns a slice of the KV cache and holds one sequence. "
            "When a request arrives the scheduler picks the slot whose cached "
            "prompt shares the longest common prefix with the incoming prompt. ")
    return ("You are a careful technical writer.\n\n"
            + body * 12
            + "\n\nTask: write a detailed explanation of how request batching works.")

def tokenize(text):
    req = urllib.request.Request(BASE + "/tokenize",
                                 data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["tokens"]

def detokenize(tokens):
    req = urllib.request.Request(BASE + "/detokenize",
                                 data=json.dumps({"tokens": tokens}).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["content"]

def build_prompt_to(n_prompt):
    """Build a prompt that tokenizes to ~n_prompt tokens (chat template adds
    a few wrapper tokens on the wire; the actual count shows up as `prompt_n`
    in jsonl rows). Reads SEED_PATH once and caches the resulting token list."""
    try:
        with open(SEED_PATH) as f:
            seed_text = f.read()
    except OSError as e:
        sys.exit(f"Could not read seed corpus at {SEED_PATH}: {e}")
    if not seed_text:
        sys.exit(f"Seed corpus at {SEED_PATH} is empty")
    toks = tokenize(seed_text)
    if not toks:
        sys.exit("/tokenize returned no tokens for the seed corpus -- server tokenizer issue?")
    if len(toks) < n_prompt:
        reps = (n_prompt + len(toks) - 1) // len(toks)
        toks = toks * reps
    return detokenize(toks[:n_prompt])

def post(prompt, n, seed, timeout=3600):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "n": n,
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,                 # force exactly MAX_TOKENS per completion
        "seed": seed,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def metrics():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception as e:
        sys.exit(f"/metrics failed ({e}). Start the server with --metrics.")
    out = {}
    for line in text.splitlines():
        if line.startswith("llamacpp:"):
            k, v = line.split()
            out[k[len("llamacpp:"):]] = float(v)
    return out

def fetch_props():
    """Fetch /props once. Returns the full dict; caller picks fields."""
    try:
        with urllib.request.urlopen(BASE + "/props", timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        sys.exit(f"Could not read {BASE}/props ({e}) -- is the server up?")

def clear_slots(n_slots):
    """Erase the prompt cache of every slot. Required setup for a 'cold' run.

    /slots/{id}?action=erase requires the server to be started with
    --slot-save-path; without that flag the endpoint returns 501. There is no
    -1-broadcast form on this endpoint -- iterate explicitly.
    """
    for sid in range(n_slots):
        req = urllib.request.Request(BASE + f"/slots/{sid}?action=erase",
                                     data=b"", method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if "slot-save-path" in body:
                sys.exit("/slots erase needs the server started with "
                         "--slot-save-path PATH (any path; erase doesn't write "
                         "to it but the flag gates the endpoint).")
            sys.exit(f"/slots/{sid}?action=erase failed: {e.code} {body}")
        except Exception as e:
            sys.exit(f"/slots/{sid}?action=erase failed: {e}")

def run_parallel(n):
    """One cold parallel request with n completions. Returns timing dict.

    Setup: erase every slot's prompt cache so the run is genuinely cold.
    """
    clear_slots(SLOTS)
    m0 = metrics(); t0 = time.time()
    r = post(PROMPT, n, SEED)
    wall = time.time() - t0
    decodes = metrics()["n_decode_total"] - m0["n_decode_total"]
    pt = r.get("timings", {})
    prompt_s = pt.get("prompt_ms", 0) / 1e3
    gen_s    = wall - prompt_s
    gen_tok  = n * MAX_TOKENS
    return dict(n=n, wall=wall, prompt_s=prompt_s, gen_s=gen_s, gen_tok=gen_tok,
                decodes=decodes, prompt_n=pt.get("prompt_n", 0),
                cache_n=pt.get("cache_n", 0))

def median(xs):
    s = sorted(xs); m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

# ============================================================================
PROPS = fetch_props()
SLOTS = int(PROPS.get("total_slots", 0))
if SLOTS <= 0:
    sys.exit(f"/props returned total_slots={SLOTS} -- is the server configured with -np?")
META.update({
    "model":       PROPS.get("model_alias") or os.path.basename(PROPS.get("model_path", "")),
    "model_path":  PROPS.get("model_path", ""),
    "build_info":  PROPS.get("build_info", ""),
    "total_slots": SLOTS,
})
print(f"server: model={META['model']!r}  build={META['build_info']!r}  total_slots={SLOTS}")

# Build the shared prompt once. With --n-prompt we tokenize a public-domain
# seed, repeat-and-truncate to the target token count, then detokenize back
# to text. Without it, the original made-up technical-writer prompt is used.
if N_PROMPT is not None:
    PROMPT = build_prompt_to(N_PROMPT)
    META["n_prompt_target"] = N_PROMPT
    META["seed_corpus"]     = os.path.basename(SEED_PATH)
    print(f"prompt: --n-prompt={N_PROMPT}  seed={os.path.basename(SEED_PATH)!r}  "
          f"(chat template adds a few wrapper tokens on the wire; actual "
          f"prompt_n shows up in each row's timings)")
else:
    PROMPT = make_default_prompt()

if SWEEP:
    nlist = [n for n in NLIST if n <= SLOTS]
    skipped = [n for n in NLIST if n not in nlist]
    print(f"SWEEP  max_tokens={MAX_TOKENS}  total_slots={SLOTS}  reps={REPS}  n values={nlist}")
    if skipped:
        print(f"  (skipping {skipped}: exceeds -np {SLOTS}; restart server with a bigger -np)")

    rows = []
    for n in nlist:
        runs = []
        for rep in range(REPS):
            r = run_parallel(n)
            r["agg_tps"]        = r["gen_tok"] / r["gen_s"]
            r["per_stream_tps"] = MAX_TOKENS / r["gen_s"]
            r["tok_per_decode"] = r["gen_tok"] / r["decodes"]
            runs.append(r)
            jdump({"mode": "sweep", "rep": rep, "reps": REPS,
                   "max_tokens": MAX_TOKENS, **r})
        agg = [r["agg_tps"] for r in runs]
        med = sorted(runs, key=lambda r: r["agg_tps"])[len(runs) // 2]  # median run
        med["agg_min"], med["agg_max"] = min(agg), max(agg)
        rows.append(med)
        spread = f"  (spread {min(agg):.1f}-{max(agg):.1f})" if REPS > 1 else ""
        print(f"  n={n:<3d} done: gen {med['gen_s']:6.1f}s  agg {med['agg_tps']:6.1f} tok/s{spread}")

    base = rows[0]["agg_tps"]
    print(f"\n{'n':>4} {'wall(s)':>9} {'prompt(s)':>10} {'gen(s)':>8} "
          f"{'tok/dec':>8} {'agg tok/s':>10} {'per-stream':>11} {'speedup':>8}")
    print("-" * 74)
    for r in rows:
        print(f"{r['n']:>4} {r['wall']:>9.2f} {r['prompt_s']:>10.2f} {r['gen_s']:>8.2f} "
              f"{r['tok_per_decode']:>8.2f} {r['agg_tps']:>10.1f} "
              f"{r['per_stream_tps']:>11.2f} {r['agg_tps']/base:>7.2f}x")
    if REPS > 1:
        print(f"\n(values are the median of {REPS} runs per n)")
    print("\nspeedup = aggregate throughput(n) / throughput(1)")
    print("        = how much faster n completions as one request is vs serial")
    sys.exit(0)

# ---- default: A/B test ------------------------------------------------------
print(f"A/B  max_tokens={MAX_TOKENS}  n={N}  (both phases cold via /slots erase)")

p = run_parallel(N)
jdump({"mode": "ab_parallel", "max_tokens": MAX_TOKENS,
       "agg_tps": p["gen_tok"] / p["gen_s"],
       "per_stream_tps": MAX_TOKENS / p["gen_s"],
       "tok_per_decode": p["gen_tok"] / p["decodes"], **p})
print(f"\n=== PARALLEL  (1 request, n={N}) ===")
print(f"  wall clock          : {p['wall']:8.2f} s")
print(f"  prompt (parent)     : {p['prompt_n']:.0f} proc / {p['cache_n']:.0f} cached"
      f"  in {p['prompt_s']:.2f} s")
print(f"  decode calls        : {p['decodes']:8.0f}   for {p['gen_tok']} gen tokens")
print(f"  tokens per decode   : {p['gen_tok']/p['decodes']:8.2f}   (~busy slots per step)")
print(f"  gen-only time       : {p['gen_s']:8.2f} s  -> {p['gen_tok']/p['gen_s']:.1f} tok/s aggregate")

clear_slots(SLOTS)  # cold req#1; intentionally NOT cleared between #2..N so they hit the cache
m0 = metrics(); t0 = time.time()
seq_prompt_s = 0.0
cache_ns, proc_ns = [], []
for i in range(N):
    rt0 = time.time()
    r = post(PROMPT, 1, SEED + i)
    rwall = time.time() - rt0
    t = r.get("timings", {})
    cache_ns.append(int(t.get("cache_n", -1)))
    proc_ns.append(int(t.get("prompt_n", -1)))
    rprompt_s = t.get("prompt_ms", 0) / 1e3
    seq_prompt_s += rprompt_s
    jdump({"mode": "ab_sequential", "req_idx": i, "n_total": N,
           "max_tokens": MAX_TOKENS, "wall": rwall,
           "prompt_s": rprompt_s, "gen_s": rwall - rprompt_s,
           "gen_tok": MAX_TOKENS,
           "prompt_n": int(t.get("prompt_n", -1)),
           "cache_n": int(t.get("cache_n", -1))})
seq_dt = time.time() - t0
seq_decodes = metrics()["n_decode_total"] - m0["n_decode_total"]
seq_gen_s   = seq_dt - seq_prompt_s
seq_gen_tok = N * MAX_TOKENS
print(f"\n=== SEQUENTIAL (separate requests x{N}, n=1) ===")
print(f"  wall clock          : {seq_dt:8.2f} s")
print(f"  prompt proc per req : {proc_ns}")
print(f"  prompt cached per req: {cache_ns}")
print(f"  (req #1 processes the prompt; #2..N reuse it -> cache_n = prompt len)")
print(f"  total prompt time   : {seq_prompt_s:8.2f} s")
print(f"  decode calls        : {seq_decodes:8.0f}   for {seq_gen_tok} gen tokens")
print(f"  tokens per decode   : {seq_gen_tok/seq_decodes:8.2f}   (~busy slots per step)")
print(f"  gen-only time       : {seq_gen_s:8.2f} s  -> {seq_gen_tok/seq_gen_s:.1f} tok/s aggregate")

print(f"\n=== SUMMARY (same {N*MAX_TOKENS} tokens generated both ways) ===")
print(f"  wall clock     : parallel {p['wall']:7.2f} s  vs  sequential {seq_dt:7.2f} s"
      f"   -> {seq_dt/p['wall']:.2f}x")
print(f"  generation only: parallel {p['gen_s']:7.2f} s  vs  sequential {seq_gen_s:7.2f} s"
      f"   -> {seq_gen_s/p['gen_s']:.2f}x")
