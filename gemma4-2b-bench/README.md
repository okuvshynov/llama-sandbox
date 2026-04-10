# Gemma4-E2B Performance Benchmark

Prefill and token generation throughput benchmark for Gemma4-E2B-it across quantization levels.

## Hardware

### Mac Pro (2019)
- **CPU**: Intel Xeon W-3245 @ 3.20GHz (16C/32T)
- **GPU**: AMD Radeon Pro Vega II Duo (4x via MoltenVK/Vulkan)

### Mac Studio M2 Ultra
- **CPU**: Apple M2 Ultra (24-core, 16P+8E)
- **GPU**: Apple M2 Ultra (76-core Metal)

## Setup

- **Tool**: `llama-bench` from llama.cpp
- **Model**: Gemma4-E2B-it (~4.6B params)
- **Quants tested**: Q2_K_XL, Q3_K_XL, Q4_K_XL, Q5_K_XL, Q6_K_XL, Q8_K_XL, BF16
- **Prefill prompt sizes**: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 tokens
- **Token generation sizes**: 16, 256, 1024 tokens
- **Measurement**: prefill (`-n 0`) and token generation (`-p 0`)

## Results

### Prefill

| File | Machine | Backend | Config |
|------|---------|---------|--------|
| `prefill-vega2-vulkan.csv` | Mac Pro 2019 | Vulkan (Vega II Duo) | `-ngl 999 --device Vulkan0` |
| `prefill-xeon-w3245-cpu.csv` | Mac Pro 2019 | CPU only | `-ngl 0` |
| `prefill-m2ultra-metal.csv` | Mac Studio M2 Ultra | Metal GPU | `-ngl 99` |
| `prefill-m2ultra-cpu.csv` | Mac Studio M2 Ultra | CPU only | `-ngl 0` |

### Token Generation

| File | Machine | Backend | Config |
|------|---------|---------|--------|
| `tg-m2ultra-metal.csv` | Mac Studio M2 Ultra | Metal GPU | `-ngl 99` |
| `tg-m2ultra-cpu.csv` | Mac Studio M2 Ultra | CPU only | `-ngl 0` |

## Scripts

- `run-vega-prefill.sh` — Mac Pro GPU (Vulkan)
- `run-cpu-prefill.sh` — Mac Pro CPU
- `run-m2ultra-gpu-prefill.sh` — M2 Ultra Metal GPU
- `run-m2ultra-cpu-prefill.sh` — M2 Ultra CPU prefill
- `run-m2ultra-cpu-decode.sh` — M2 Ultra CPU token generation
