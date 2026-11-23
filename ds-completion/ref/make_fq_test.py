#!/usr/bin/env python3
"""Inputs + expected outputs of kernel_cpu's fake-quant functions, for bin/test-fakequant.

Rows of 64 values at magnitudes from 1e-7 to 1e4, including exact ties of the e2m1 grid and (since step 192) of the e4m3 grid,
zeros, and blocks whose amax hits the clamps. Writes tests/data/fq/{in,fp8,fp4,fp4e4}.f32 (in git since README step 186: 182 KB each).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import kernel_cpu as K  # noqa: E402

torch.manual_seed(0)
rows = []
for mag in [1e-7, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 3, 10, 100, 1e4]:
    for _ in range(64):
        rows.append(torch.randn(64) * mag)
rows.append(torch.zeros(64))
# exact grid midpoints (ties) at power-of-two scales: e2m1 ties 0.25, 0.75, ..., 5 and e4m3 ties
ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 4)
for sc in [2.0**-3, 1.0, 2.0**5]:
    rows.append(torch.cat([ties, -ties]) * sc)
# e4m3 ties under the fp8 scale (README step 192: the rows above hold none, so a wrong tie rule passed
# the fp8 comparison). Each block of 32 has amax 448 * sc, so its scale is sc; then odd multiples
# of half a step in several binades: 17 = between 16 and 18, 4.25 between 4 and 4.5, ...
e4 = torch.tensor([448.0, 17, 19, 21, 23, 25, 27, 29, 31, 34, 38, 42, 46, 4.25, 4.75, 5.25, 5.75, 1.0625, 1.1875, 1.3125,
                   100, 108, 116, 124, 200, 216, 232, 248, 0.53125, 0.59375, 2.125, 2.375])
for sc in [1.0, 2.0**-4]:
    rows.append(torch.cat([e4, -e4]) * sc)
x = torch.stack(rows).float()
out = Path("tests/data/fq")
out.mkdir(parents=True, exist_ok=True)
x.numpy().tofile(out / "in.f32")
K.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True).numpy().tofile(out / "fp8.f32")
K.fp4_act_quant(x.clone(), 32, True).numpy().tofile(out / "fp4.f32")
K.fp4_act_quant(x.clone(), 16, True, torch.float8_e4m3fn).numpy().tofile(out / "fp4e4.f32")
print(f"{x.shape[0]} rows x 64 -> {out}")
