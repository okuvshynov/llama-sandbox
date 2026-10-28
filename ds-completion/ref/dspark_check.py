#!/usr/bin/env python3
"""The official DSpark draft (model.py forward_spec) fed with *our* main hidden states, to tell a
difference in the draft head from a difference in what the main model hands it.

    ~/venvs/ds-ref/bin/python ref/dspark_check.py data/ref/x-dspark.pt ours.bin --dspark DIR

ours.bin and ours.bin.hidden come from ds-test --e2e ... --dspark-out ours.bin; x-dspark.pt
from ds_ref.py --dspark (its token sequence and anchors). Runs in the fp32 + fake-quant variant.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds_ref  # noqa: E402  (patches model.py on import)
from ds_ref import GGUF, HF_DIR, GGUFEngramEmbedding, TokenizerAdapter, install_lazy_experts, load_dense, ref  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ref_pt")
ap.add_argument("ours")
ap.add_argument("--dspark", required=True)
ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args()

R = torch.load(a.ref_pt, weights_only=False)
steps = R["steps"]
n_prompt = len(R["prompt_ids"])
torch.set_default_dtype(torch.float32)
torch.set_num_threads(a.threads)
cfg = json.load(open(HF_DIR / "inference/config.json"))
args = ref.ModelArgs(**cfg)
args.max_batch_size, args.max_seq_len = 1, 1024
args.temperature = 0.0
args.vision_n_layers = 0
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
_emb_init = ref.ParallelEngramEmbedding.__init__


def emb_init(self, num_embeddings, dim):
    with torch.device("meta"):
        _emb_init(self, num_embeddings, dim)


ref.ParallelEngramEmbedding.__init__ = emb_init
model = ref.Transformer(args, TokenizerAdapter(tok))
ref.ParallelEngramEmbedding.__init__ = _emb_init
G, Gd = GGUF(), GGUF(Path(a.dspark))
base = 0
for L, rows in zip(args.engram_layer_ids, args.engram_num_embeddings):
    model.layers[L].engram.embed = GGUFEngramEmbedding(G, base, rows, args.engram_head_dim)
    base += rows
load_dense(model, G, Gd)
for p in model.parameters():
    if p.dtype == torch.bfloat16:
        p.data = p.data.float()
install_lazy_experts(model, G, Gd, args.n_layers)

E = args.dim
n_t = len(args.dspark_target_layer_ids)
hidden = torch.from_numpy(np.fromfile(a.ours + ".hidden", dtype="<f4").reshape(-1, n_t * E))
B = args.dspark_block_size
V = args.vocab_size
rec = np.dtype([("ids", "<i4", (B + 1,)), ("conf", "<f4", (B,)), ("logits", "<f4", (B, V))])
ours = np.fromfile(a.ours, dtype=rec)
print(f"# {len(hidden)} hidden positions, {len(ours)} drafts of ours")

rec = {}


def keep(key, t):  # hooks must return None (a pre-hook's return value replaces the arguments)
    rec.setdefault(key, t.clone())


for l, blk in enumerate(model.mtp):
    blk.attn.register_forward_pre_hook(lambda m, a, l=l: keep((l, 0), a[0]))
    blk.attn.register_forward_hook(lambda m, a, o, l=l: keep((l, 1), o))
    blk.ffn.register_forward_pre_hook(lambda m, a, l=l: keep((l, 2), a[0]))
    blk.ffn.register_forward_hook(lambda m, a, o, l=l: keep((l, 3), o))
dbg_path = Path(a.ours + ".hidden.debug")
with torch.inference_mode():
    # prefill: seed the draft windows with the prompt positions
    model.forward_spec(torch.tensor([steps[0]["next"]]), hidden[:n_prompt].unsqueeze(0), 0)
    for k in range(1, len(steps)):
        pos = n_prompt + k - 1
        anchor = torch.tensor([steps[k]["next"]])
        if k == 1:
            rec.clear()
        ids, logits, conf = model.forward_spec(anchor, hidden[pos].view(1, 1, -1), pos)
        if k == 1 and dbg_path.exists():
            n_l = len(model.mtp)
            d = np.fromfile(dbg_path, dtype="<f4").reshape(n_l, 4, B, E)
            names = ["attn_in", "attn_out", "ffn_in", "ffn_out"]
            for l in range(n_l):
                row = []
                for w in range(4):
                    r_ = rec[(l, w)][0].float().numpy()
                    row.append(f"{names[w]} {np.linalg.norm(d[l, w] - r_) / np.linalg.norm(r_):.1e}")
                print(f"  first draft, layer {l}: " + "  ".join(row))
                for w in range(4):
                    r_ = rec[(l, w)][0].float().numpy()
                    per = [np.linalg.norm(d[l, w, j] - r_[j]) / np.linalg.norm(r_[j]) for j in range(B)]
                    print(f"      {names[w]:8s} per position " + " ".join(f"{x:.1e}" for x in per))
        lg = logits[0].numpy()
        o = ours[k - 1]
        err = [np.linalg.norm(o["logits"][j] - lg[j]) / np.linalg.norm(lg[j]) for j in range(B)]
        rl = steps[k]["draft_logits"].numpy()
        err_ref = [np.linalg.norm(rl[j] - lg[j]) / np.linalg.norm(lg[j]) for j in range(B)]
        print(f"step {k:2d}: official draft on our hidden {ids[0].tolist()}  ours {list(o['ids'])}  "
              f"relerr vs ours {max(err):.1e}  vs reference run {max(err_ref):.1e}")
