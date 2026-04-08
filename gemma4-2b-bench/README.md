# Gemma4-E2B Prefill Performance Benchmark

Prompt processing (prefill) throughput benchmark for Gemma4-E2B-it across quantization levels, run on a Mac Pro (2019).

## Hardware

- **CPU**: Intel Xeon W-3245 @ 3.20GHz (16C/32T)
- **GPU**: AMD Radeon Pro Vega II Duo (4x via MoltenVK/Vulkan)

## Setup

- **Tool**: `llama-bench` from llama.cpp (build `69c28f15`, #8696)
- **Model**: Gemma4-E2B-it (~4.6B params)
- **Quants tested**: Q2_K_XL, Q3_K_XL, Q4_K_XL, Q5_K_XL, Q6_K_XL, Q8_K_XL, BF16
- **Prompt sizes**: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 tokens
- **Measurement**: prompt processing only (`-n 0 -d 0`)

## Results

| File | Backend | Config |
|------|---------|--------|
| `prefill-vega2-vulkan.csv` | Vulkan (Vega II Duo, device 0) | `-ngl 999 --device Vulkan0` |
| `prefill-xeon-w3245-cpu.csv` | CPU only | `-ngl 0` |

## Scripts

- `run-vega-prefill.sh` -- GPU run (all layers offloaded to Vulkan0)
- `run-cpu-prefill.sh` -- CPU-only run (no GPU layers)
