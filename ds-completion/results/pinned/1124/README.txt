# Pinned-only loads, the reserve, and the model on huge pages: measurements of 2026-11-24 (README step 203).
# Commit 042e890 (engine unchanged: a scratch build with scratch-build.diff); sclk 6 (manual, all four cards); booted 2026-11-24 08:26;
# kernel 6.10.5-1-t2-jammy; ROCm 6.3.4.
#
# scratch-build.diff   the scratch build's environment switches. Used here: DS_CACHE_PINNED (no registration, every load through
#                      the pinned buffer, loads from the first token) and DS_RESERVE_GB (rt.h's reserve_gb). The others are step 201-202's.
# heldout/             scripts/run_matrix.py <heldout/matrix.txt> --keep-out: 8 held-out prompts x 4 configurations x 2 repeats, the records.
#                      registered = today's fastest settings (--cache-wait); pinned = DS_CACHE_PINNED=1; pinned200 = the same at
#                      --cache-loads 200; nowait = registered without --cache-wait. All 64 outputs equal results/cache-test/plain-canon-static/.
# reserve/             the same prompts: pinned against pinned with DS_RESERVE_GB=0.3 (2 repeats). All 32 outputs equal the stored ones.
# long/                tests/data/long-prompt.txt (2,731 tokens), -n 100, the base arguments of heldout/matrix.txt with --cache-loads 800,
#                      -f in place of -p, by hand, twice: registered (--cache-wait), pinned, pinned-r03, nowait. The 8 outputs are one text.
# reg_huge.cpp         registers 6 GiB on 2 MB pages of the hugetlb pool (a memfd, shared, read-only) or on the page cache    out/huge.out
# gguf_huge.cpp        llama.cpp's GGUF reader on a file in the pool                                                         out/huge.out
# out/decode_huge.out  the CPU expert kernel on 4 KB against 2 MB pages (bench-cpu-moe --decode)
# Build lines: reg_huge as results/register/1124/README.txt's first; gguf_huge:
#   g++ -O2 -std=c++17 -fopenmp -I$LLAMA_CPP/ggml/include gguf_huge.cpp -o gguf_huge $LLAMA_CPP/build/ggml/src/libggml-base.a -lpthread
# Read the records: scripts/records.py heldout/*-2.json --by config --show counters.gen_tps
#
# Generation, t/s, second repeat (the first: 44.75 / 41.63 / 39.86 / 29.41)
    prompt        registered     pinned  pinned200     nowait
    tcp-udp            39.02      36.45      37.74      30.04
    py-lru             46.63      44.70      41.57      32.85
    cpp-matrix         54.85      50.68      45.01      33.44
    sql-report         47.29      42.86      41.28      30.09
    geometry           55.03      51.08      47.13      34.54
    zh-travel          33.94      33.30      32.00      21.20
    ja-email           37.29      36.96      34.26      20.90
    summary-json       45.09      39.54      40.42      33.26
    mean               44.89      41.95      39.93      29.54      (tests/data/speed/baseline.tsv: 44.88)
    vs registered                 -6.6%     -11.1%     -34.2%
    uses on GPUs       84.6%      84.0%      79.7%      60.4%
    loads published    3,330      3,353      1,463          0
    verify steps, s     5.65       6.02       6.34       9.14
    drafts, s           0.69       0.70       0.72       0.74
    loaded in, s        75.1       12.8       12.8       12.8
#
# The reserve (pinned loads; experts placed 6,097 = 802 + 3 x 1,765 against 6,193 = 826 + 3 x 1,789)
                         0.75 GB      0.3 GB
    mean, repeat 1         42.39       42.48   (+0.2%)
    mean, repeat 2         42.53       42.39   (-0.3%)
    uses on GPUs, rep 2   84.10%      84.37%
#
# The long prompt (two runs each)
                    prompt t/s       generation t/s   uses on GPUs   loads published   loaded in
    registered      89.00  89.70     30.84  33.02     71.7% 71.6%    3,080  2,857      75 s
    pinned          88.44  89.50     32.74  32.42     71.6% 71.6%    2,845  2,831      13 s
    pinned-r03      89.87  90.25     31.95  32.15     72.2% 72.2%    2,931  2,917      13 s
    nowait          75.68  75.17     24.00  23.94     64.4% 64.4%        0      0      13 s
#
# Huge pages (out/huge.out): registering 6 GiB, GB/s
                                  page cache (this boot)   2 MB pages of the pool      289 GB at the pool's rate
    hipHostRegister, 4 dies       2.9                      7.8, 8.6                    34-37 s
    hsa_amd_memory_lock, 4 dies   7.0                      11.3, 13.3                  22-26 s
    hsa_amd_memory_lock, 3 dies   -                        16.0                        18 s
    hsa_amd_memory_lock, 1 die    -                        23.0                        13 s
    HIP's copy to die 1 from the registered memory: 12.8-12.9 GB/s on both, the bytes equal.
    The pool filled from the page cache by 16 threads: 31-33 GB/s.
# The cards' used memory (mem_info_vram_used, MB, dies in sysfs order) with 6 GiB registered for 4 dies (./reg_huge $M4 <mode> 8, read after 6 s):
    hsa4 (the pool):        10 10 10 10  ->  12 10 10 10
    cache-hsa4:             10 10 10 10  ->  23 21 21 21
# The GGUF reader takes shard 1 in the pool (padded to 2 MB: 92 keys, the same offsets). Shard 1 has no tensors; a shard with
# tensors (47 GiB) does not fit an 8 GiB pool and was not tried. Reserving 474 GiB was not tried either.
# The CPU kernel (out/decode_huge.out): 74-86 GB/s on 4 KB pages, 76-90 on 2 MB pages: no difference beyond the rows' scatter.
