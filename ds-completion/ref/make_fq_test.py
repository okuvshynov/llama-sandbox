#!/usr/bin/env python3
"""Inputs + expected outputs of kernel_cpu's fake-quant functions, for bin/test-fakequant.

Rows of 64 values at magnitudes from 1e-7 to 1e4, including exact ties of the e4m3 / e2m1 grids,
zeros, and blocks whose amax hits the clamps. Writes data/fq/{in,fp8,fp4,fp4e4}.f32.
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
x = torch.stack(rows).float()
out = Path("data/fq")
out.mkdir(parents=True, exist_ok=True)
x.numpy().tofile(out / "in.f32")
K.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True).numpy().tofile(out / "fp8.f32")
K.fp4_act_quant(x.clone(), 32, True).numpy().tofile(out / "fp4.f32")
K.fp4_act_quant(x.clone(), 16, True, torch.float8_e4m3fn).numpy().tofile(out / "fp4e4.f32")
print(f"{x.shape[0]} rows x 64 -> {out}")
