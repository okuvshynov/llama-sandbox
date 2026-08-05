# logit-kld — logit collection for cross-model KL divergence

Utilities for measuring KL divergence between two models — potentially different
quantizations, different settings, or entirely different inference frameworks.

The workflow is split in two so tokenizer/chat-template differences can't
contaminate the model comparison:

1. **`collect`** (this project, llama.cpp-based): greedy continuation of a prompt,
   recording the token id sequence plus per-position top-K logits and a full-vocab
   log-sum-exp normalizer.
2. **Rescoring utilities** (future, one per inference framework): consume the *raw
   token ids* from a collect file, score the same sequence under a second
   model/framework, and emit the same file format. The KL comparison then pairs two
   files position by position. At that step we evaluate the model itself — the
   token sequence is fixed, so tokenizer or template bugs in either framework
   don't change what's being compared.

## Why top-K + normalizer

Storing the full vocab per position (~150k floats) is wasteful for
storage/transfer. Instead each position stores its top-K raw logits (default
K=128) plus the full-vocab log-sum-exp, split as `max_logit + lse_rest`:

- exact probabilities for the stored entries: `p_k = exp(logit_k − max_logit − lse_rest)`;
- the residual tail mass `1 − Σ p_k` is exact, and it upper-bounds the error of a
  KL computed over the top-K only (truncated KL underestimates the true KL).

The `max_logit`/`lse_rest` split keeps precision: `lse_rest ∈ [0, ln n_vocab]`, so
both floats carry full relative precision, whereas one fp32 log-sum-exp at
magnitude ~30 would inject absolute error on the order of small quant KLs.
Logits are fp32 exactly as llama.cpp produced them (a position is ~1KB at K=128;
size is a non-issue at this scale).

## Prerequisites

- C++17 compiler, CMake ≥ 3.14
- llama.cpp source tree (built as part of this project via `add_subdirectory`)
- Python 3 for `inspect.py` (stdlib only)

## Build

```bash
cmake -B build -DLLAMA_CPP_DIR=$HOME/projects/llama.cpp
cmake --build build -j
```

## Usage

```bash
./build/collect -m model.gguf -p "The capital of France is" -n 256 -k 128 -o run.bin

# multi-shard models: pass the first shard, llama.cpp picks up the rest
./build/collect -m GLM-5.2-UD-Q6_K-00001-of-00014.gguf -f prompt.txt -o glm.bin
```

Flags: `-o` output (default `logits.bin`), `-n` tokens to generate (256), `-k`
top-K stored (128), `-c` context size (4096, auto-raised to fit), `-b` decode
chunk size (512), `-ngl` GPU layers (0), `-t` threads (all cores).

Generation is greedy only — the next token is by construction the stored top-1 of
the previous position, which `inspect.py` verifies end to end. The completion text
streams to stdout; a summary (tok/s, stop reason, tail-mass stats — the signal for
whether K is large enough) goes to stderr.

**No chat template is ever applied.** The prompt is tokenized raw with
`add_special=true, parse_special=true`: the model's BOS is honored, and literal
special-token text in the prompt (e.g. a pre-rendered template from any external
framework) is parsed into special tokens. The recorded token ids are the ground
truth interface regardless of how the prompt text was produced.

Memory note: llama.cpp allocates an output buffer of `chunk × n_vocab` floats per
decode call — at a 150k vocab the default `-b 512` costs ~300MB; raising `-b`
raises this linearly.

## Inspecting output

```bash
python inspect.py run.bin              # header, per-seq stats, sanity checks
python inspect.py run.bin --dump-tokens
```

`inspect.py` is also the format's **reference reader** — future rescoring
utilities should parse files the way its `read_file()` does. Checks: structural
consistency, `max_logit == top logit`, descending sort, `lse_rest` range, top-K
mass ≤ 1, and greedy self-consistency (`ids[0]` at position i equals token i+1
over the generated range).

## File format (`lkldtopk` v1)

Little-endian, packed, magic `"lkldtopk"`. Strings are `uint32 len` + UTF-8 bytes.

```
Header:
  char   magic[8]  = "lkldtopk"
  uint32 version   = 1
  int32  n_vocab
  int32  top_k                 # K actually stored = min(requested K, n_vocab)
  int32  n_seq                 # sequences in file (currently 1 per collect run)
  string model_desc            # model path + llama_model_desc()

Per sequence (× n_seq):
  string label                 # prompt source ("inline" or file path)
  int32  n_prompt              # prompt token count
  int32  n_total               # prompt + generated
  int32  n_scored              # positions with logit records (= n_total in v1)
  int32  tokens[n_total]
  per position p in 0..n_scored-1:
    float  max_logit           # max over full vocab (== topk_logits[0])
    float  lse_rest            # log Σ_v exp(logit_v − max_logit) over full vocab
    int32  topk_ids[top_k]     # sorted by logit desc, ties by id asc
    float  topk_logits[top_k]  # raw fp32 logits
```

Every position is scored, including position 0 and the last one — position i's
distribution predicts token i+1, and prompt positions are included so a
downstream KL covers the prompt and the completion alike. Downstream tools can
ignore positions (e.g. the final one, which predicts beyond the sequence) as they
see fit; `n_scored` is explicit so the convention isn't hardcoded.

Output `.bin` files are gitignored (root `.gitignore`); the magic bytes identify
the format.
