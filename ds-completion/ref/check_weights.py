#!/usr/bin/env python3
"""Prove the GGUF is the original checkpoint, tensor by tensor, and that our name mapping is right.

    ~/venvs/ds-ref/bin/python ref/check_weights.py [--layers 0,1,2,20] [--experts 8|all] [--engram-rows 4096]

For every GGUF tensor of the selected layers (plus the global ones) the GGUF bytes are decoded with
ggml's block semantics and compared for exact float equality against the safetensors decoded with
the reference's semantics (see weights.py). Also checked:
  - every HF tensor of those layers is used by some GGUF tensor (nothing silently dropped);
  - the hyper-parameters in the GGUF match config.json;
  - the engram hash constants in the GGUF (primes, offsets, multipliers, compressed token map) match
    what the reference engram.py computes from scratch.
The default source is the 1017 checkpoint on Hugging Face (the GGUF's actual source), read with
HTTP range requests; ~/llms/ds-flash is the base release, whose weights differ slightly.
Routed experts are sampled (first, last and random ones) unless --experts all; the 189 GiB engram
table is checked on 3 contiguous runs of 1024 rows (start/middle/end of each layer's table) plus random rows.
Exit status is non-zero if anything differs.
"""
import argparse
import functools
import json
import random
import re
import sys
import time

import torch

from weights import EXPS, GGUF, HF, HF_DIR, LAYER_MAP, HFRemote, fp4_rows, fp8_rows, hf_value

print = functools.partial(print, flush=True)
failures = []


def compare(label, g: torch.Tensor, h: torch.Tensor):
    if g.shape != h.shape:
        failures.append(f"{label}: shape {tuple(g.shape)} vs {tuple(h.shape)}")
        print(f"  FAIL {label}: shape {tuple(g.shape)} vs {tuple(h.shape)}")
        return
    if torch.isnan(g).any() or torch.isnan(h).any():
        failures.append(f"{label}: NaN")
        print(f"  FAIL {label}: NaN present")
        return
    if not torch.equal(g, h):
        d = (g - h).abs()
        failures.append(f"{label}: max |diff| {d.max().item():.3e}, {(d > 0).sum().item()} elements differ")
        print(f"  FAIL {label}: max |diff| {d.max().item():.3e}, {(d > 0).sum().item()} of {d.numel()} differ")
        return
    print(f"  ok   {label:44s} {str(tuple(g.shape)):22s} exact")


def check_hparams(G: GGUF):
    print("== hyper-parameters (GGUF vs config.json)")
    c = json.load(open(HF_DIR / "config.json"))["text_config"]
    a = "deepseek41."
    pairs = [
        ("block_count", c["num_hidden_layers"]), ("embedding_length", c["hidden_size"]),
        ("attention.head_count", c["num_attention_heads"]), ("attention.key_length", c["head_dim"]),
        ("rope.dimension_count", c["qk_rope_head_dim"]), ("attention.q_lora_rank", c["q_lora_rank"]),
        ("attention.output_group_count", c["o_groups"]), ("attention.output_lora_rank", c["o_lora_rank"]),
        ("attention.sliding_window", c["sliding_window"]), ("attention.compress_ratios", c["compress_ratios"]),
        ("attention.compress_rope_freq_base", c["compress_rope_theta"]), ("rope.freq_base", c["rope_theta"]),
        ("rope.scaling.factor", c["rope_scaling"]["factor"]),
        ("rope.scaling.original_context_length", c["rope_scaling"]["original_max_position_embeddings"]),
        ("rope.scaling.yarn_beta_fast", c["rope_scaling"]["beta_fast"]),
        ("rope.scaling.yarn_beta_slow", c["rope_scaling"]["beta_slow"]),
        ("attention.layer_norm_rms_epsilon", c["rms_norm_eps"]),
        ("expert_count", c["n_routed_experts"]), ("expert_used_count", c["num_experts_per_tok"]),
        ("expert_shared_count", c["n_shared_experts"]), ("expert_feed_forward_length", c["moe_intermediate_size"]),
        ("expert_weights_scale", c["routed_scaling_factor"]), ("expert_weights_norm", c["norm_topk_prob"]),
        ("swiglu_clamp_exp", [c["swiglu_limit"]] * c["num_hidden_layers"]),
        ("swiglu_clamp_shexp", [c["swiglu_limit"]] * c["num_hidden_layers"]),
        ("attention.indexer.head_count", c["index_n_heads"]), ("attention.indexer.key_length", c["index_head_dim"]),
        ("attention.indexer.top_k", c["index_topk"]),
        ("attention.kv_source_layers", c["kv_source_layer_ids"]),
        ("attention.indexer.source_layers", c["index_source_layer_ids"]),
        ("attention.indexer.candidate_source_layer", c["candidate_source_layer_id"]),
        ("attention.indexer.candidate_top_k_blocks", c["candidate_topk_blocks"]),
        ("attention.indexer.candidate_block_size", c["candidate_block_size"]),
        ("hyper_connection.count", c["hc_mult"]), ("hyper_connection.sinkhorn_iterations", c["hc_sinkhorn_iters"]),
        ("hyper_connection.epsilon", c["hc_eps"]),
        ("ple.layers", c["engram_layer_ids"]), ("ple.ngram_size", c["engram_max_ngram_size"]),
        ("ple.heads_per_ngram", c["engram_n_heads"]), ("embedding_length_per_layer_input", c["engram_head_dim"]),
        ("engram.vocab_size", c["engram_vocab_size"]), ("engram.pad_token_id", c["engram_pad_token_id"]),
        ("engram.compressed_vocab_size", c["engram_compressed_vocab_size"]),
    ]
    for key, want in pairs:
        got = G.field(a + key)
        same = (abs(got - want) <= 1e-6 * abs(want)) if isinstance(want, float) else got == want
        if isinstance(want, list) and isinstance(want[0], float):
            same = all(abs(x - y) <= 1e-6 * abs(y) for x, y in zip(got, want)) and len(got) == len(want)
        if not same:
            failures.append(f"hparam {key}: gguf {got} vs config {want}")
        shown = want if not isinstance(want, list) or len(want) < 10 else f"[{len(want)} values]"
        print(f"  {'ok  ' if same else 'FAIL'} {key:42s} {shown}")
    # the score function and the vocabulary are not plain keys
    gating = G.field(a + "expert_gating_func")
    print(f"  info expert_gating_func = {gating} (config: {c['scoring_func']})")
    n_vocab = len(G.field("tokenizer.ggml.token_type"))
    ok = n_vocab == c["vocab_size"]
    if not ok:
        failures.append(f"vocab {n_vocab} vs {c['vocab_size']}")
    print(f"  {'ok  ' if ok else 'FAIL'} {'vocab size':42s} {n_vocab}")


def check_engram_constants(G: GGUF):
    print("== engram hash constants (GGUF vs reference engram.py recomputed)")
    sys.path.append(str(HF_DIR / "inference"))
    from types import SimpleNamespace

    import engram as ref  # the reference module itself
    from tokenizers import Tokenizer

    c = json.load(open(HF_DIR / "config.json"))["text_config"]
    args = SimpleNamespace(engram_layer_ids=c["engram_layer_ids"], engram_max_ngram_size=c["engram_max_ngram_size"],
                           engram_n_heads=c["engram_n_heads"], engram_vocab_size=c["engram_vocab_size"],
                           engram_num_embeddings=c["engram_num_embeddings"], engram_head_dim=c["engram_head_dim"])
    layout = ref.EngramLayout.from_args(args)
    a = "deepseek41."
    primes = [p for layer in layout.primes for per in layer for p in per]
    offsets, base = [], 0
    for li, layer in enumerate(layout.primes):
        flat = [p for per in layer for p in per]
        assert sum(flat) == c["engram_num_embeddings"][li]
        offsets += [base + sum(flat[:i]) for i in range(len(flat))]  # GGUF: absolute rows in the concatenated table
        base += sum(flat)
    mult = ref.compute_hash_multipliers(tuple(c["engram_layer_ids"]), c["engram_max_ngram_size"],
                                        c["engram_compressed_vocab_size"]).flatten().tolist()
    for key, want in [("ple.head_vocab_sizes", primes), ("ple.head_offsets", offsets), ("ple.layer_multipliers", mult)]:
        got = G.field(a + key)
        ok = got == want
        if not ok:
            failures.append(f"engram {key}")
        print(f"  {'ok  ' if ok else 'FAIL'} {key:42s} {len(want)} values")

    tok = Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))

    class Adapter:  # the two members build_compressed_token_map uses
        backend_tokenizer = tok

        def __len__(self):
            return tok.get_vocab_size(with_added_tokens=True)

    t0 = time.time()
    token_map, cvocab = ref.build_compressed_token_map(Adapter())
    got = G.field(a + "engram.token_map")
    ok = token_map == got and cvocab == c["engram_compressed_vocab_size"]
    if not ok:
        n_diff = sum(x != y for x, y in zip(token_map, got)) if len(token_map) == len(got) else -1
        failures.append(f"engram token map: {n_diff} entries differ, compressed vocab {cvocab}")
    print(f"  {'ok  ' if ok else 'FAIL'} {'engram.token_map':42s} {len(token_map)} -> {cvocab} ids "
          f"({time.time() - t0:.1f}s)")


def check_layer(G: GGUF, H: HF, L: int, n_experts: str, rng: random.Random):
    print(f"== layer {L}")
    gp, hp = f"blk.{L}.", f"layers.{L}."
    n_exp = G.ne(gp + "ffn_gate_exps.weight")[2]
    ids = list(range(n_exp)) if n_experts == "all" else sorted(
        {0, n_exp - 1, *rng.sample(range(1, n_exp - 1), int(n_experts))})
    # everything this layer compares, fetched up front and concurrently (remote source)
    expert_re = re.compile(r"\.ffn\.experts\.(\d+)\.")
    H.prefetch([n for n in H.map if n.startswith(hp) and ".engram.embed." not in n
                and (not expert_re.search(n) or int(expert_re.search(n).group(1)) in ids)])
    used = set()
    for name in sorted(n for n in G.tensors if n.startswith(gp)):
        suffix = name[len(gp):]
        if suffix in EXPS:
            w = EXPS[suffix]
            rows = G.ne(name)[1]
            bad = 0
            for e in ids:
                g = G.rows(name, e * rows, (e + 1) * rows)
                h = fp4_rows(H.raw(f"{hp}ffn.experts.{e}.{w}.weight"), H.raw(f"{hp}ffn.experts.{e}.{w}.scale"))
                if g.shape != h.shape or not torch.equal(g, h):
                    bad += 1
                    failures.append(f"{name} expert {e} differs")
            for e in range(n_exp):
                used.update({f"{hp}ffn.experts.{e}.{w}.weight", f"{hp}ffn.experts.{e}.{w}.scale"})
            print(f"  {'ok  ' if bad == 0 else 'FAIL'} {suffix:44s} {len(ids)} of {n_exp} experts "
                  f"{'exact' if bad == 0 else f'{bad} differ'}")
            continue
        compare(suffix, G.tensor(name), hf_value(H, name))
        if suffix == "engram_gate.weight":
            used.update({hp + "engram.q_weight", hp + "engram.k_weight"})
        else:
            src = hp + LAYER_MAP[suffix]
            used.update({src, src + ".weight", src + ".scale"})
    # every HF tensor of this layer must be accounted for (the engram table is checked separately)
    hf_names = {n for n in H.map if n.startswith(hp) and ".engram.embed." not in n}
    missing = sorted(hf_names - used)
    if missing:
        failures.append(f"layer {L}: HF tensors not in the GGUF: {missing}")
        print(f"  FAIL unused HF tensors: {missing}")
    else:
        print(f"  ok   all {len(hf_names)} HF tensors of the layer are used")


def check_engram_table(G: GGUF, H: HF, n_rows: int, rng: random.Random):
    print("== engram table (per_layer_token_embd = layer tables concatenated)")
    c = json.load(open(HF_DIR / "config.json"))["text_config"]
    base = 0
    total = G.ne("per_layer_token_embd.weight")[1]
    for L, rows in zip(c["engram_layer_ids"], c["engram_num_embeddings"]):
        # three contiguous runs (start, middle, end of the layer's table) + random single rows
        run = 1024
        ranges = [(0, run), (rows // 2, rows // 2 + run), (rows - run, rows)]
        ranges += [(r, r + 1) for r in rng.sample(range(rows), n_rows)]
        ws = H.rows(f"layers.{L}.engram.embed.weight", ranges)
        ss = H.rows(f"layers.{L}.engram.embed.scale", ranges)
        bad = n_checked = 0
        for (r0, r1), w, s in zip(ranges, ws, ss):
            g = G.rows("per_layer_token_embd.weight", base + r0, base + r1)
            n_checked += r1 - r0
            if not torch.equal(g, fp8_rows(w, s)):
                bad += 1
        if bad:
            failures.append(f"engram table layer {L}: {bad} row ranges differ")
        print(f"  {'ok  ' if bad == 0 else 'FAIL'} layer {L}: rows {base}..{base + rows - 1}, "
              f"{n_checked} rows checked {'exact' if bad == 0 else f'({bad} ranges differ)'}")
        base += rows
    ok = base == total
    if not ok:
        failures.append(f"engram table rows {total} vs sum {base}")
    print(f"  {'ok  ' if ok else 'FAIL'} table has {total} rows = sum of layer tables")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1,2,3,14,20,21,24,39")
    ap.add_argument("--experts", default="8", help="random experts per tensor besides first/last, or 'all'")
    ap.add_argument("--engram-rows", type=int, default=512, help="random single rows per engram layer, besides 3 runs of 1024")
    ap.add_argument("--skip-constants", action="store_true")
    ap.add_argument("--hf", default="remote", choices=["remote", "local"],
                    help="remote: the 1017 checkpoint the GGUF came from, fetched by HTTP range (default); "
                         "local: ~/llms/ds-flash, the base release (expected to differ slightly)")
    a = ap.parse_args()
    torch.set_num_threads(16)
    rng = random.Random(0)
    t0 = time.time()
    G = GGUF()
    H = HFRemote() if a.hf == "remote" else HF()
    print(f"# {len(G.tensors)} GGUF tensors in {len(G.shards)} shards, {len(H.map)} HF tensors")
    check_hparams(G)
    if not a.skip_constants:
        check_engram_constants(G)
    print("== global tensors")
    for name in ["token_embd.weight", "output.weight", "output_norm.weight"]:
        compare(name, G.tensor(name), hf_value(H, name))
    check_engram_table(G, H, a.engram_rows, rng)
    layers = range(G.field("deepseek41.block_count")) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    for L in layers:
        check_layer(G, H, L, a.experts, rng)
    if isinstance(H, HFRemote):
        print(f"# fetched {H.bytes_fetched / 1e9:.2f} GB from {H.url}")
    print(f"\n{'ALL OK' if not failures else f'{len(failures)} FAILURES'} ({time.time() - t0:.0f}s)")
    for f in failures:
        print("  " + f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
