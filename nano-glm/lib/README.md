# lib — the engine the apps share

Everything in here is model-and-mechanism; nothing decides policy. Apps in
`../apps` supply the policy: what to read, what to emit, when to stop.

| file | what |
|---|---|
| `nano_model.h` | hparams, GGUF shard loader, read-only mapping |
| `nano_graph.h` | backends, KV cache, the glm-dsa trunk graph, one chunk eval |
| `moe_block.h` | the routed-expert graph — router, top-k, expert FFNs, combine |
| `moe_client.h` | the trunk's side of the remote MoE seam: connect, handshake, stats, the custom-op callback |
| `moe_proto.h` | the client/backend wire protocol and the TCP it needs |
| `build_info.h` | build fingerprint: `--version`, provenance, handshake |
| `expert_trace.h` | routing trace, compiled out unless `-DNANO_EXPERT_TRACE` |
| `vocab.h` | byte-level BPE: GGUF vocab and merges, the glm4 pre-tokenizer, encode/decode |
| `chat_glm.h` | GLM-5.2's single-turn chat format, as token ids |
| `unicode_ranges.h` | generated `\p{L}` / `\p{N}` tables — see `gen_unicode_ranges.py` |

Include `nano_graph.h` first in any app: it reaches `moe_proto.h`, and
winsock2.h has to precede the windows.h that `nano_model.h` pulls in.

`../../logit-kld/src` comes along on the include path for `logits_file.{h,cpp}`
(the lkldtopk format), `cpu_topology.h` and `topk_utils.h`.

## How model-specific is any of this?

Not evenly. Counting mentions of `glm-dsa`, `_mla`, `indexer` and `hadamard`
puts the whole of it in two files:

| file | glm-dsa refs | MLA/DSA refs | scope |
|---|--:|--:|---|
| `build_info.h` | 0 | 0 | **any program** |
| `unicode_ranges.h` | 0 | 0 | **any program** |
| `moe_proto.h` | 1 | 0 | any MoE backend |
| `moe_client.h` | 0 | 0 | any MoE backend |
| `expert_trace.h` | 0 | 0 | any MoE with a top-k selection tensor |
| `moe_block.h` | 0 | 0 | **DeepSeek-lineage** MoE |
| `vocab.h` | 0 | 0 | any `gpt2` vocab; the *splitter* is glm4's |
| `nano_model.h` | 34 | 33 | half generic loader, half glm-dsa definition |
| `nano_graph.h` | 3 | 90 | **glm-dsa only** |
| `chat_glm.h` | — | — | **glm-dsa only**, and the most so of anything here |

Three tiers, then, and they are worth naming because they predict cost:

- **generic** — `build_info.h`, the protocol and client, the trace, and inside
  `nano_model.h` the GGUF key helpers, `map_file_ro`, shard splitting and the
  tensor map. Nothing here knows what a model is.
- **model family** — `moe_block.h`. Sigmoid gating, selection bias, weight
  normalisation, expert scale: that is the DeepSeek lineage, shared by
  glm-dsa, DeepSeek and Kimi. A Mixtral-style softmax-top-k router would not
  reuse it.
- **one model** — `nano_hparams`, `nano_layer`, `load_hparams`, `build_graph`,
  and the KV cache layout in `nano_state` (576-wide MLA rows, 128-wide indexer
  rows). This is where MLA absorption, the lightning indexer, the Hadamard
  rotation and the DSA top-k mask live.

## If we add a second model

Say Kimi-K3, the model this whole plan exists for. A `models/` directory would
hold **only the third tier**:

```
models/
  glm_dsa/         hparams + tensor names + build_graph + KV layout
  kimi_k3/         the same four things, written again
```

and `lib/` would keep the other two tiers plus the eval driver — backend setup,
graph reuse on `(n_tokens, n_kv)`, `eval_chunk`, `pad_n_kv`.

**Reimplement:** hparams and their asserts (~200 lines), tensor names,
`build_graph` (~300 lines), the KV cache shape, and the chat format. Roughly
500 lines plus a template.

The tokenizer is a smaller question than it looks: `vocab.h`'s BPE serves any
`tokenizer.ggml.model == "gpt2"` vocab unchanged, and a model declaring a
different `tokenizer.ggml.pre` needs only a new `pretok_split` — one function,
because that is all a pre-tokenizer is. Both `load_vocab` checks are hard
aborts precisely so an unported combination cannot be mistaken for a working
one.

**Reuse unchanged:** the loader plumbing, `moe_block.h` if the gating matches,
the whole RPC stack including `moe-server`, the trace, the fingerprint, and —
the part that actually costs time to build — `gate.py`, the lkldtopk format,
and `compare.py`.

**Copy, do not abstract.** Two `build_graph`s that share a few `ggml_mul_mat`
calls are two functions; a `build_graph` behind an interface that both models
implement is a framework, and the reason nano-glm exists is that llama.cpp
became one. The op order in a graph *is* the model, and making it configurable
makes it unreadable. Reach for a shared abstraction on the third model, not the
second, and only for the part that repeated identically twice.

### The two things that decide the real cost

**Gating.** The RPC contract sends one activation and gets one combined row
back, which means the router, the selection bias, the weight norm and the
scale are all backend knowledge (PLAN.md "Goal"). A model in the DeepSeek
lineage slots in for free. One with different gating forces a choice: a second
code path in the server, or moving the router back to the client and paying
`n_expert_used` rows per request instead of one. Decide that before writing the
graph, not after.

**Attention shape.** `nano_state` hardcodes two caches with MLA and
lightning-indexer geometry. A model with plain MHA/GQA, or without a sparse
indexer, needs a different cache — that is the one piece of `lib/` outside the
third tier that a second model is likely to disturb, and the honest fix is to
move the cache description into the model module rather than to parameterise it.

### Also per-model, outside the code

`testdata/` is a golden set for *one model on one machine* — new model, new
`prompts.json` and a fresh `gate.py --update-golden`. `gate.py` itself needs no
change; `DEFAULT_MODEL` is already an argument.

## Header-only, for now

`nano-lib` is a CMake INTERFACE target: these are headers with `static`
functions, compiled into each app rather than linked from an archive. That is
fine while every app is a single translation unit, and it keeps the build
honest — one compile, no archive, no link order.

**It stops being fine the moment an app has two translation units.** A `static`
definition in a header gives each TU its own copy, so anything holding state —
`g_moe` and `g_rpc_ctxs` in the client, `g_expert_trace` — would silently
become two independent instances, and the failure would look like the trace
losing half its rows or the RPC counters reading low, not like a linkage bug.

So: the first app that needs a second `.cpp` must convert this to a real static
library first — the state-holding pieces move to `.cpp` files with declarations
here, and the pure functions can stay inline. Do not paper over it by marking
things `inline`; that fixes the ODR problem and leaves the design question
unanswered.

## What must not change casually

`load_model()` builds `M.desc` as `"glm-dsa %uL nano-glm"` and every app writes
it into its output file's header. The committed golden set in `../testdata`
contains that string, so changing it changes the bytes and fails the gate for
reasons that have nothing to do with the model. Rename it only alongside a
`gate.py --update-golden`, and say so in the commit.

The same goes for the op order in `moe_block.h` and the graph in the trunk:
they mirror llama.cpp op for op, and the resemblance is the acceptance test.
