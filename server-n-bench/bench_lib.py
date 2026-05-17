"""
Shared helpers for the bench_*.py scripts (bench_n.py, bench_sequential.py).

Each script is a one-shot measurement: it makes a single primitive request
shape against a running llama.cpp server, optionally appends a JSONL row,
and exits. Sweeps and repetitions are driven by shell loops.

Server prerequisites:
  -np <max n>  --metrics  --slot-save-path PATH

--slot-save-path is required because it gates the entire /slots/* action
endpoint (server-context.cpp:3572). The /slots/{id}?action=erase calls
this module issues don't actually write to PATH, but the flag must be set.

NOTE on /metrics prompt_tokens_total: for n>1 the server copies the parent
slot's prompt-token count into every child slot (copy_state_to,
server-context.cpp:562), so that counter reports ~n x the real work.
Use each response's `timings` block for real prompt cost; only
n_decode_total from /metrics is correct here and used for decode-call counts.
"""
import json
import os.path
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8080"
SEED = 1234

# Seed corpus for --n-prompt. Default is seed_corpus.cpp (canonical
# algorithms in C++) -- representative of a coding workload, which matters
# for MoE models where expert routing depends on input content. Override
# via env var BENCH_N_SEED.
SEED_PATH = os.environ.get(
    "BENCH_N_SEED",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "seed_corpus.cpp"),
)


def _install_user_agent():
    # Some proxies in front of remote llama-server tunnels (runpod,
    # cloudflare) reject the stdlib's default "Python-urllib/3.x" with
    # HTTP 403 while letting curl through. addheaders are defaults;
    # explicit headers on a Request (e.g. Content-Type) still win.
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "bench_n.py/0.1")]
    urllib.request.install_opener(opener)


_install_user_agent()


def parse_common_args(argv):
    """Pull --jsonl PATH, --n-prompt N, --base-url URL from any position in argv.

    Returns (remaining_argv, jsonl_path, n_prompt, base_url). https:// works
    as-is via urllib's system CA bundle.
    """
    jsonl_path, n_prompt, base = None, None, DEFAULT_BASE
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "--jsonl" and i + 1 < len(argv):
            jsonl_path = argv[i + 1]; i += 2
        elif argv[i] == "--n-prompt" and i + 1 < len(argv):
            n_prompt = int(argv[i + 1]); i += 2
        elif argv[i] == "--base-url" and i + 1 < len(argv):
            base = argv[i + 1].rstrip("/"); i += 2
        else:
            out.append(argv[i]); i += 1
    return out, jsonl_path, n_prompt, base


def _read_seed():
    try:
        with open(SEED_PATH) as f:
            text = f.read()
    except OSError as e:
        sys.exit(f"Could not read seed corpus at {SEED_PATH}: {e}")
    if not text:
        sys.exit(f"Seed corpus at {SEED_PATH} is empty")
    return text


def tokenize(base, text):
    req = urllib.request.Request(base + "/tokenize",
                                 data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["tokens"]


def detokenize(base, tokens):
    req = urllib.request.Request(base + "/detokenize",
                                 data=json.dumps({"tokens": tokens}).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["content"]


def build_prompt_to(base, n_prompt):
    """Build a prompt that tokenizes to ~n_prompt tokens via /tokenize on
    the seed corpus, repeat-and-truncate, then /detokenize back to text.

    The chat template adds a few wrapper tokens on the wire; the actual
    count shows up as `prompt_n` in jsonl rows.
    """
    toks = tokenize(base, _read_seed())
    if not toks:
        sys.exit("/tokenize returned no tokens for the seed corpus -- server tokenizer issue?")
    if len(toks) < n_prompt:
        reps = (n_prompt + len(toks) - 1) // len(toks)
        toks = toks * reps
    return detokenize(base, toks[:n_prompt])


def post(base, prompt, n, seed, max_tokens, timeout=3600):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "n": n,
        "max_tokens": max_tokens,
        "ignore_eos": True,                 # force exactly max_tokens per completion
        "seed": seed,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def metrics(base):
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception as e:
        sys.exit(f"/metrics failed ({e}). Start the server with --metrics.")
    out = {}
    for line in text.splitlines():
        if line.startswith("llamacpp:"):
            k, v = line.split()
            out[k[len("llamacpp:"):]] = float(v)
    return out


def fetch_props(base):
    try:
        with urllib.request.urlopen(base + "/props", timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        sys.exit(f"Could not read {base}/props ({e}) -- is the server up?")


def clear_slots(base, n_slots):
    """Erase every slot's prompt cache. Required setup for a 'cold' run.

    /slots/{id}?action=erase requires the server started with
    --slot-save-path; without it the endpoint returns 501. There is no
    -1-broadcast form -- iterate explicitly.
    """
    for sid in range(n_slots):
        req = urllib.request.Request(base + f"/slots/{sid}?action=erase",
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


class BenchContext:
    """Bundles base URL, meta (for jsonl rows), the shared prompt, and the
    target JSONL path. Both bench scripts construct this via setup()."""

    def __init__(self, base, jsonl_path, meta, prompt):
        self.base = base
        self.jsonl_path = jsonl_path
        self.meta = meta
        self.prompt = prompt
        self.slots = int(meta["total_slots"])

    def clear_slots(self):
        clear_slots(self.base, self.slots)

    def metrics(self):
        return metrics(self.base)

    def post(self, n, seed, max_tokens):
        return post(self.base, self.prompt, n, seed, max_tokens)

    def jdump(self, row):
        if self.jsonl_path is None:
            return
        row = {"ts": time.time(), **self.meta, **row}
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")


def setup(argv):
    """Parse common args, fetch /props, build the shared prompt, print a
    banner, and return (BenchContext, remaining_positional_argv)."""
    remaining, jsonl_path, n_prompt, base = parse_common_args(argv)
    props = fetch_props(base)
    slots = int(props.get("total_slots", 0))
    if slots <= 0:
        sys.exit(f"/props returned total_slots={slots} -- is the server configured with -np?")
    meta = {
        "base_url":    base,
        "model":       props.get("model_alias") or os.path.basename(props.get("model_path", "")),
        "build_info":  props.get("build_info", ""),
        "total_slots": slots,
    }
    if n_prompt is not None:
        prompt = build_prompt_to(base, n_prompt)
        meta["n_prompt_target"] = n_prompt
    else:
        prompt = _read_seed()
    print(f"server: base={base}  model={meta['model']!r}  build={meta['build_info']!r}  "
          f"total_slots={slots}")
    if n_prompt is not None:
        print(f"prompt: --n-prompt={n_prompt}  (chat template adds a few wrapper "
              f"tokens; actual prompt_n shows up in each row)")
    else:
        print(f"prompt: seed file used as-is (no --n-prompt)")
    return BenchContext(base, jsonl_path, meta, prompt), remaining
