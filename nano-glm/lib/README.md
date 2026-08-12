# lib — the engine the apps share

Everything in here is model-and-mechanism; nothing decides policy. Apps in
`../apps` supply the policy: what to read, what to emit, when to stop.

| file | what |
|---|---|
| `gguf_store.h` | GGUF metadata helpers, read-only mapping, shard enumeration, the tensor map |
| `moe_shape.h` | the MoE dimensions a client and a backend must agree on |
| `moe_client.h` | the trunk's side of the remote MoE seam: connect, handshake, stats, the custom-op callback |
| `moe_proto.h` | the client/backend wire protocol and the TCP it needs |
| `build_info.h` | build fingerprint: `--version`, provenance, handshake |
| `expert_trace.h` | routing trace, compiled out unless `-DNANO_EXPERT_TRACE` |
| `vocab.h` | byte-level BPE: GGUF vocab and merges, the glm4 pre-tokenizer, encode/decode |
| `unicode_ranges.h` | generated `\p{L}` / `\p{N}` tables — see `gen_unicode_ranges.py` |
| `prompt_source.h` | prompt token ids from an lkldtopk file or a literal list |

and one directory per architecture, holding everything that knows which model
it is looking at:

| file | what |
|---|---|
| `models/glm_dsa/model.h` | GLM-5.2 hparams, tensor names, loader |
| `models/glm_dsa/graph.h` | backends, KV cache, the glm-dsa trunk graph, one chunk eval |
| `models/glm_dsa/moe_block.h` | its routed-expert graph, as a router half and an expert half plus the composition the trunk calls |
| `models/glm_dsa/chat.h` | GLM-5.2's single-turn chat format, as token ids |
| `models/deepseek4/model.h` | DeepSeek-V4-Flash hparams, tensor names, loader |
| `models/deepseek4/moe_block.h` | its routed-expert graph, same two halves, plus hash routing and an optional naming hook |
| `models/deepseek4/graph.h` | its trunk graph — hyper-connections, MLA attention at all three compression ratios, the lightning indexer, the FFN half. Every layer; still no head, and it aborts past what has been checked |

Include the model's `graph.h` first in any app: it reaches `moe_proto.h`, and
winsock2.h has to precede the windows.h that `gguf_store.h` pulls in.

`../../logit-kld/src` comes along on the include path for `logits_file.{h,cpp}`
(the lkldtopk format), `cpu_topology.h` and `topk_utils.h`.

## How model-specific is any of this?

Not evenly. Counting mentions of `glm-dsa`, `_mla`, `indexer` and `hadamard`
puts the whole of it in two files:

Two tiers in `lib/`, and a third under `models/`:

- **generic** — `gguf_store.h` (GGUF key helpers, `map_file_ro`, shard
  splitting, the tensor map), `build_info.h`, the protocol and client, the
  trace, `unicode_ranges.h`, `prompt_source.h`. Nothing here knows what a model
  is.
- **shared contract** — `moe_shape.h`. Not ops, just the dimensions a client
  and a backend must agree on, which is exactly what `moe_hello_response`
  already carried over the wire.
- **one model** — everything under `models/<arch>/`: hparams and their asserts,
  tensor names, the routed-expert graph, the trunk graph, the KV layout, the
  chat format.

## Adding a second model: what it actually cost

DeepSeek-V4-Flash (`deepseek4`) is the second architecture, and it is worth
scoring the prediction this section used to make.

**Right:** `models/` holds only the third tier; `lib/` keeps the rest.
Hparams, tensor names, graph and KV layout are indeed the four things that get
written again. "Copy, do not abstract" held — see below.

**Wrong, and instructively:** the old text said `moe_block.h` was a *family*
tier, reusable across "the DeepSeek lineage" if the gating matched. It did not
match. deepseek4 gates with `sqrt(softplus(x))` where glm-dsa uses sigmoid, and
clamps its SwiGLU. Two models in the same lineage, and the ops still differ.

So the family tier was wishful: what is genuinely shared between two MoE models
is the *shape* — `n_expert`, `n_expert_used`, `n_ff_exp`, the scale and the
norm flag — and not a line of arithmetic. That is now `moe_shape.h`, which is
small and honest, and each architecture writes its own expert graph.

The **seam** between the halves, on the other hand, held exactly. `moe-server`'s
device machinery — host-side routing, per-device compaction, the combine — is
written against `{ids, weights}` in and activations out, and deepseek4 dropped
into it without the server learning anything about sqrt-softplus, clamped
SwiGLU or hash routing. Two architectures whose arithmetic shares nothing still
share their *decomposition*, which is the tier boundary that turned out to be
real.

**One latent bug surfaced**, which is the usual dividend of a second case:
`load_shard` built a CPU buffer for every shard including metadata-only ones,
and `gguf_get_data_offset` returns the *unpadded* header end when a shard has
no tensors. GLM-5.2's metadata shard happens to end on a multiple of 32;
DeepSeek's ends 18 bytes past one, and ggml asserted. Nothing was wrong with
the second model.

**Reuse unchanged:** the loader plumbing (after that fix), the whole RPC stack
including `moe-server`, the trace, the fingerprint, and — the part that
actually costs time to build — `gate.py`, the lkldtopk format, and
`compare.py`.

The tokenizer is a smaller question than it looks: `vocab.h`'s BPE serves any
`tokenizer.ggml.model == "gpt2"` vocab unchanged, and a model declaring a
different `tokenizer.ggml.pre` needs only a new `pretok_split` — one function,
because that is all a pre-tokenizer is. Both `load_vocab` checks are hard
aborts precisely so an unported combination cannot be mistaken for a working
one.

**On the cost estimate:** the old text guessed ~500 lines. Hparams, tensor
names and the loader alone came to ~330 for deepseek4, and the trunk graph is
the larger half and not yet written — hyper-connections replace the residual
stream with four Sinkhorn-mixed streams, so it is not a matter of swapping an
attention kernel. Expect the estimate to be right for a model in the same
shape as one already ported, and low for one that is not.

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
