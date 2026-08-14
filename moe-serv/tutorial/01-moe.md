# 1. What a Mixture-of-Experts model is

## The decode loop, in one paragraph

A transformer language model generates text one token at a time. To produce
each token it runs the current sequence through a stack of layers; each layer
is an **attention** block (which lets the new token look at earlier ones)
followed by a **feed-forward network** (FFN — two or three big matrix
multiplications with a nonlinearity). The output of the last layer becomes a
probability distribution over the vocabulary; one token is sampled; repeat.
This per-token phase is called **decode**. Processing the initial prompt,
where many tokens can be pushed through together, is called **prefill** —
same math, very different performance character (more on that below).

## The FFN, and the MoE idea

In a dense transformer the FFN is the same weights for every token. In a
**Mixture-of-Experts (MoE)** transformer, each layer instead has many
independent FFNs — the **experts** — plus a small **router**. Per token, the
router scores all experts and only the top-k actually run; their outputs are
combined, weighted by the router's scores.

In DeepSeek-V4-Flash, each MoE layer has **256 experts** and each token
activates **6** of them (plus one always-on **shared expert** that every
token uses). So the model has enormous capacity — 256 experts' worth of
weights per layer — while each token only pays for 6.

The catch is where those weights live. In this model the routed experts are
**137 of the 150.7 GiB** — over 90% of the model. Everything else (attention,
the shared expert, embeddings, the output head — collectively "the trunk") is
about 13.7 GiB.

## Why decode is a memory problem, not a compute problem

For one token, each expert's job is a vector-times-matrix multiply: a
4096-element input against a 4096×2048 matrix, twice (called *gate* and
*up*), then 2048×4096 back down (*down*). Each weight is read from memory,
used in exactly **one** multiply-add, and never touched again for this token.
There is no reuse — so the speed limit is not how fast you can multiply, but
how fast you can *read weights from memory*. This is what "memory-bound"
means, and it is the single most important fact about MoE decode.

Concretely, per layer per token: 6 experts × 3 matrices × ~4.5 MB ≈ **80 MB
of expert weights read**. Times 43 MoE layers ≈ 3.4 GB read per token. At
this machine's CPU memory bandwidth (~75 GB/s achieved), that alone is tens
of milliseconds — and indeed the whole model decodes at ~3.6 tokens/s on the
CPU, ~280 ms per token.

Prefill is different: with 512 prompt tokens in flight, each expert that gets
loaded is used by many tokens at once, so the same weight read is amortized
across many multiplies. Prefill is compute-bound; decode is bandwidth-bound.
Every optimization in this project is judged separately for the two, because
what helps one often does nothing for the other
([06](06-optimization.md)).

## Which part is worth accelerating

A decode step on this machine spends roughly 20% of its time in the routed
expert block and 80% in the trunk. So even making the expert block *free*
only buys ~1.25x here — a fact worth internalizing before celebrating any
expert-block speedup. The project's answer
([`PLAN.md`](../PLAN.md#goal)): the trunk is small (13.7 GiB) and would fit
on one modern GPU, where it becomes fast; the expert block is the part that
is *hard to place anywhere* because of its size, so making it fast on cheap,
large-memory hardware is the piece of the puzzle worth owning. On this
machine, after the work in this tutorial, the block runs 3.1x faster than
the CPU and MoE is down to ~11% of decode time.

## Terms to carry forward

| term | meaning here |
|---|---|
| decode | generating tokens one at a time; bandwidth-bound |
| prefill | processing the prompt in big batches; compute-bound |
| expert | one independent FFN (three matrices) inside a MoE layer |
| router | tiny network choosing which 6 of 256 experts a token uses |
| shared expert | one extra FFN every token always uses |
| trunk | everything that is not the routed experts (~13.7 GiB) |
| memory-bound | performance limited by bytes/second, not math/second |
