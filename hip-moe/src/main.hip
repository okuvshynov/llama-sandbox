// hip-moe — standalone HIP/ROCm benchmarks for the moe-serv MoE kernels.
//
// Same hardware as the Windows/Vulkan work (Mac Pro 7,1, four Vega II dies,
// gfx906), different OS/driver/toolchain: Linux + ROCm. This reruns the shapes
// for which moe-serv has recorded Vulkan numbers (moe-serv/docs/KERNEL.md,
// vk-latency/README.md) so the two stacks can be compared die for die:
//
//   matmul gate/up  k=4096 m=2048 x6 experts   Vulkan 95.8 us (ggml-vk 163.8)
//   matmul down     k=2048 m=4096 x6 experts   Vulkan 88.9 us
//   tp block (fused GU + SwiGLU + down)        Vulkan 113 us/die
//   latency (null kernel)                      Vulkan: submit 9 us, round ~67,
//                                              4-die round 87 us
//
// Deliberately standalone: no ggml. MXFP4 blocks are generated synthetically
// (random nibbles + bounded e8m0 scales) and the CPU reference dequantizes
// them with the same semantics the kernels use (ggml-common.h kvalues_fp4,
// ggml-impl.h ggml_e8m0_to_fp32_half, copied exactly) — GPU and reference
// consume the same bytes and differ only in summation order, so the gate is
// the probe's: |diff| <= 1e-4 + 1e-3|ref|. A speedup on a MISMATCH row does
// not exist.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

// ---------------------------------------------------------------------------
// MXFP4 host side: ggml's 17-byte block, bit for bit (ggml-common.h): one
// e8m0 scale byte, then 16 nibble bytes where qs[j] holds elements j (low)
// and j+16 (high) of a 32-element sub-block.
// ---------------------------------------------------------------------------

struct block_mxfp4_raw {
    uint8_t e;
    uint8_t qs[16];
};

// kvalues_fp4: doubled-e2m1, bit 3 the sign. Index 8 is -0 == 0.
static const float KVALUES[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

static float e8m0_half_host(uint8_t e) {
    const uint32_t bits = e < 2 ? (0x00200000u << e) : (((uint32_t) e - 1u) << 23);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// Random quantized experts: nibbles uniform, scales e8m0 in [117,123]
// (2^-11..2^-5 after the half), giving weights of rms ~0.1 — the magnitude
// regime of the Vulkan probe, whose weights were ggml-quantized from
// uniform[-1,1] (scale ~= rowmax/12). Larger scales fail the probe's
// tolerance gate honestly: outputs near zero built from +-hundreds of
// fp32 partials carry ~1e-3 of cancellation dust, which the |diff| <=
// 1e-4 + 1e-3|ref| contract does not cover.
static std::vector<uint8_t> random_blocks(std::mt19937 & rng, int64_t k, int64_t m) {
    const int64_t kb = k / 32;
    std::vector<uint8_t> blocks((size_t) (m * kb * 17));
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_int_distribution<int> escale(117, 123);
    for (int64_t mm = 0; mm < m; mm++) {
        block_mxfp4_raw * row = (block_mxfp4_raw *) (blocks.data() + mm * kb * 17);
        for (int64_t b = 0; b < kb; b++) {
            row[b].e = (uint8_t) escale(rng);
            for (int j = 0; j < 16; j++) row[b].qs[j] = (uint8_t) byte(rng);
        }
    }
    return blocks;
}

// Dequantize one expert into f32 [m][k] row-major (row mm = output column mm).
static void dequant(const uint8_t * blocks, int64_t k, int64_t m, float * out) {
    const int64_t kb = k / 32;
    for (int64_t mm = 0; mm < m; mm++) {
        const block_mxfp4_raw * row = (const block_mxfp4_raw *) (blocks + mm * kb * 17);
        float * dst = out + mm * k;
        for (int64_t b = 0; b < kb; b++) {
            const float sc = e8m0_half_host(row[b].e);
            for (int j = 0; j < 16; j++) {
                dst[b * 32 + j]      = KVALUES[row[b].qs[j] & 0x0F] * sc;
                dst[b * 32 + j + 16] = KVALUES[row[b].qs[j] >> 4]   * sc;
            }
        }
    }
}

// The repack (moe-serv/src/moe_probe_custom.cpp, verbatim): 17-byte blocks ->
// two planes.
//   nibble plane u32[K][M/8]: entry (k, t) = columns 8t..8t+7 at row k,
//                             k-paired for uvec2 loads (E2)
//   scale plane  u8[K/32][M]: (k-block, column)
static void repack(const uint8_t * blocks, int64_t k, int64_t m,
                   std::vector<uint32_t> & plane, std::vector<uint8_t> & scales) {
    const int64_t kb = k / 32;
    plane.assign((size_t) (k * m / 8), 0);
    scales.assign((size_t) (kb * m), 0);
    for (int64_t mm = 0; mm < m; mm++) {
        const block_mxfp4_raw * row = (const block_mxfp4_raw *) (blocks + mm * kb * 17);
        for (int64_t b = 0; b < kb; b++) {
            scales[(size_t) (b * m + mm)] = row[b].e;
            for (int j = 0; j < 16; j++) {
                const uint32_t lo = row[b].qs[j] & 0x0F;
                const uint32_t hi = row[b].qs[j] >> 4;
                const int64_t k_lo = b * 32 + j;
                const int64_t k_hi = b * 32 + j + 16;
                plane[(size_t) ((((k_lo >> 1) * (m / 8) + mm / 8) << 1) | (k_lo & 1))] |= lo << (4 * (mm % 8));
                plane[(size_t) ((((k_hi >> 1) * (m / 8) + mm / 8) << 1) | (k_hi & 1))] |= hi << (4 * (mm % 8));
            }
        }
    }
}

// Slice nrows rows starting at row0 into a plane laid out for m_total columns,
// placing them at column dst_row0 (the fused-GU repack). plane/scales must be
// pre-sized and zeroed by the caller.
static void repack_slice(const uint8_t * blocks, int64_t k,
                         int64_t row0, int64_t nrows,
                         int64_t m_total, int64_t dst_row0,
                         std::vector<uint32_t> & plane, std::vector<uint8_t> & scales) {
    const int64_t kb = k / 32;
    for (int64_t r = 0; r < nrows; r++) {
        const int64_t mm = dst_row0 + r;
        const block_mxfp4_raw * row = (const block_mxfp4_raw *) (blocks + (row0 + r) * kb * 17);
        for (int64_t b = 0; b < kb; b++) {
            scales[(size_t) (b * m_total + mm)] = row[b].e;
            for (int j = 0; j < 16; j++) {
                const uint32_t lo = row[b].qs[j] & 0x0F;
                const uint32_t hi = row[b].qs[j] >> 4;
                const int64_t k_lo = b * 32 + j;
                const int64_t k_hi = b * 32 + j + 16;
                plane[(size_t) ((((k_lo >> 1) * (m_total / 8) + mm / 8) << 1) | (k_lo & 1))] |= lo << (4 * (mm % 8));
                plane[(size_t) ((((k_hi >> 1) * (m_total / 8) + mm / 8) << 1) | (k_hi & 1))] |= hi << (4 * (mm % 8));
            }
        }
    }
}

// Slice a block-aligned k-range of every row: down's split, where die d owns
// input rows [I*d, I*(d+1)) = blocks [I*d/32, ...).
static void repack_kslice(const uint8_t * blocks, int64_t k_src, int64_t m,
                          int64_t kb0, int64_t nkb,
                          std::vector<uint32_t> & plane, std::vector<uint8_t> & scales) {
    const int64_t kb_src = k_src / 32;
    for (int64_t mm = 0; mm < m; mm++) {
        const block_mxfp4_raw * row = (const block_mxfp4_raw *) (blocks + mm * kb_src * 17);
        for (int64_t b = 0; b < nkb; b++) {
            const block_mxfp4_raw & blk = row[kb0 + b];
            scales[(size_t) (b * m + mm)] = blk.e;
            for (int j = 0; j < 16; j++) {
                const uint32_t lo = blk.qs[j] & 0x0F;
                const uint32_t hi = blk.qs[j] >> 4;
                const int64_t k_lo = b * 32 + j;
                const int64_t k_hi = b * 32 + j + 16;
                plane[(size_t) ((((k_lo >> 1) * (m / 8) + mm / 8) << 1) | (k_lo & 1))] |= lo << (4 * (mm % 8));
                plane[(size_t) ((((k_hi >> 1) * (m / 8) + mm / 8) << 1) | (k_hi & 1))] |= hi << (4 * (mm % 8));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernels: line-for-line ports of moe-serv/shaders/*.comp (the E2+E6 winner).
// ---------------------------------------------------------------------------

// Doubled-e2m1 magnitudes for nibble&7, packed 4 bits each: {0,1,2,3,4,6,8,12}.
#define TBL 0xC8643210u

__device__ __forceinline__ float e8m0_half(uint32_t e) {
    const uint32_t bits = e < 2u ? (0x00200000u << e) : ((e - 1u) << 23);
    return __uint_as_float(bits);
}

// Pass 1: partial dot products per K-tile. E6 thread shape: two threads share
// one u32 column group, four columns each.
__global__ __launch_bounds__(256) void k_pass1(
        const uint2    * __restrict__ w,     // nibble plane, k-paired
        const uint32_t * __restrict__ s,     // scale plane, u8 as u32
        const float    * __restrict__ x,     // [K] (or [S][K] when x_stride=K)
        float          * __restrict__ p,     // [S][KT][M]
        const int32_t  * __restrict__ ids,   // [S] expert per slot
        uint32_t K, uint32_t M, uint32_t tile_k, uint32_t n_tiles,
        uint32_t w_stride, uint32_t s_stride, uint32_t x_stride) {
    __shared__ float xs[128];
    __shared__ float lut[16];

    const uint32_t tid      = threadIdx.x;
    const uint32_t gtid     = blockIdx.x * 256u + tid;
    const uint32_t tcol     = gtid >> 1u;              // which u32 column group
    const uint32_t half_sel = (gtid & 1u) * 16u;       // which 16-bit half of it
    const uint32_t kt       = blockIdx.y;              // which K-tile
    const uint32_t slot     = blockIdx.z;              // which routed slot
    const uint32_t e        = (uint32_t) ids[slot];    // which expert's weights

    if (tid < 16u) {
        const float mag = (float) ((TBL >> (4u * (tid & 7u))) & 0xFu);
        lut[tid] = (tid & 8u) != 0u ? -mag : mag;
    }
    const uint32_t k0 = kt * tile_k;
    for (uint32_t i = tid; i < tile_k; i += 256u) {
        xs[i] = x[slot * x_stride + k0 + i];
    }
    __syncthreads();

    const uint32_t m8 = M / 8u;
    if (tcol >= m8) return;
    const uint32_t col0 = tcol * 8u + (half_sel >> 2u);

    const uint32_t wbase = e * w_stride + tcol;
    const uint32_t sbase = e * s_stride;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    for (uint32_t sb = 0u; sb < tile_k / 32u; sb++) {
        const uint32_t kb = k0 / 32u + sb;
        float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;

        for (uint32_t j = 0u; j < 16u; j++) {
            const uint32_t dk = sb * 32u + 2u * j;
            const float x0 = xs[dk];
            const float x1 = xs[dk + 1u];
            const uint2 wv = w[wbase + ((k0 + dk) >> 1u) * m8];
            const uint32_t hx = wv.x >> half_sel, hy = wv.y >> half_sel;
            b0 += x0 * lut[(hx >>  0u) & 0xFu] + x1 * lut[(hy >>  0u) & 0xFu];
            b1 += x0 * lut[(hx >>  4u) & 0xFu] + x1 * lut[(hy >>  4u) & 0xFu];
            b2 += x0 * lut[(hx >>  8u) & 0xFu] + x1 * lut[(hy >>  8u) & 0xFu];
            b3 += x0 * lut[(hx >> 12u) & 0xFu] + x1 * lut[(hy >> 12u) & 0xFu];
        }

        const uint32_t sc = s[sbase + (kb * M + col0) / 4u];
        a0 += e8m0_half((sc >>  0u) & 0xFFu) * b0;
        a1 += e8m0_half((sc >>  8u) & 0xFFu) * b1;
        a2 += e8m0_half((sc >> 16u) & 0xFFu) * b2;
        a3 += e8m0_half((sc >> 24u) & 0xFFu) * b3;
    }

    const uint32_t pbase = ((slot * n_tiles + kt) * M) + col0;
    p[pbase + 0u] = a0;
    p[pbase + 1u] = a1;
    p[pbase + 2u] = a2;
    p[pbase + 3u] = a3;
}

// Pass 2: reduce the K-tile partials into the output.
__global__ __launch_bounds__(256) void k_pass2(
        const float * __restrict__ p, float * __restrict__ y,
        uint32_t M, uint32_t n_tiles) {
    const uint32_t col = blockIdx.x * 256u + threadIdx.x;
    const uint32_t e   = blockIdx.z;
    if (col >= M) return;

    float acc = 0.0f;
    const uint32_t base = e * n_tiles * M + col;
    for (uint32_t t = 0u; t < n_tiles; t++) {
        acc += p[base + t * M];
    }
    y[e * M + col] = acc;
}

// TP stage 2: reduce fused gate|up partials, clamp both halves, SwiGLU.
__global__ __launch_bounds__(256) void k_tp_mid(
        const float * __restrict__ p, float * __restrict__ h,
        uint32_t inter, uint32_t n_tiles,
        float gate_min, float gate_max, float up_min, float up_max) {
    const uint32_t c    = blockIdx.x * 256u + threadIdx.x;
    const uint32_t slot = blockIdx.z;
    if (c >= inter) return;

    const uint32_t m2   = 2u * inter;
    const uint32_t base = slot * n_tiles * m2;
    float g = 0.0f, u = 0.0f;
    for (uint32_t t = 0u; t < n_tiles; t++) {
        g += p[base + t * m2 + c];
        u += p[base + t * m2 + inter + c];
    }
    g = fminf(fmaxf(g, gate_min), gate_max);
    u = fminf(fmaxf(u, up_min),   up_max);
    const float sg = g / (1.0f + expf(-g));   // ggml_swiglu_split: silu(gate) * up
    h[slot * inter + c] = sg * u;
}

// TP stage 4: reduce the down partials and apply the router weight.
__global__ __launch_bounds__(256) void k_tp_final(
        const float * __restrict__ p, const float * __restrict__ wt,
        float * __restrict__ y, uint32_t M, uint32_t n_tiles) {
    const uint32_t c    = blockIdx.x * 256u + threadIdx.x;
    const uint32_t slot = blockIdx.z;
    if (c >= M) return;

    float acc = 0.0f;
    const uint32_t base = slot * n_tiles * M + c;
    for (uint32_t t = 0u; t < n_tiles; t++) {
        acc += p[base + t * M];
    }
    y[slot * M + c] = acc * wt[slot];
}

__global__ void k_null() {}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

template <typename T>
static T * to_device(const std::vector<T> & v) {
    T * d = nullptr;
    HIP_CHECK(hipMalloc(&d, v.size() * sizeof(T)));
    HIP_CHECK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static void print_kernel_info(const void * fn, const char * name) {
    hipFuncAttributes a;
    HIP_CHECK(hipFuncGetAttributes(&a, fn));
    printf("  %-10s vgprs %d  lds %zu B  spill %zu B\n",
           name, a.numRegs, a.sharedSizeBytes, a.localSizeBytes);
}

// Combined tolerance gate, same as the Vulkan probe.
static bool check(const std::vector<float> & y, const std::vector<double> & ref) {
    double max_abs = 0.0, max_excess = 0.0, at_y = 0.0, at_ref = 0.0;
    int64_t at = -1;
    for (size_t i = 0; i < y.size(); i++) {
        const double d = fabs((double) y[i] - ref[i]);
        if (d > max_abs) { max_abs = d; at = (int64_t) i; at_y = y[i]; at_ref = ref[i]; }
        const double excess = d - (1e-4 + 1e-3 * fabs(ref[i]));
        if (excess > max_excess) max_excess = excess;
    }
    const bool ok = max_excess <= 0.0;
    printf("  check vs f32-dequant reference: max abs %.3e  %s\n", max_abs, ok ? "ok" : "MISMATCH");
    if (!ok) printf("  worst at %lld: got %+.6e  ref %+.6e\n", (long long) at, at_y, at_ref);
    return ok;
}

// --- matmul: the KERNEL.md standard measurement -----------------------------

static int run_matmul(int64_t k, int64_t m, int64_t n_used, int reps, int64_t tile_k, int dev) {
    if (k % 32 || m % 8 || tile_k % 32 || tile_k > 128) {
        fprintf(stderr, "need k%%32==0, m%%8==0, tile_k%%32==0 and <=128\n");
        return 2;
    }
    HIP_CHECK(hipSetDevice(dev));

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<uint8_t>> expert_blocks(n_used);
    std::vector<uint32_t> plane_all;
    std::vector<uint8_t> scales_all;
    {
        std::vector<uint32_t> plane;
        std::vector<uint8_t> scales;
        for (int64_t e = 0; e < n_used; e++) {
            expert_blocks[e] = random_blocks(rng, k, m);
            repack(expert_blocks[e].data(), k, m, plane, scales);
            plane_all.insert(plane_all.end(), plane.begin(), plane.end());
            scales_all.insert(scales_all.end(), scales.begin(), scales.end());
        }
    }
    // scale plane is read as u32 — pad to a multiple of 4 bytes.
    while (scales_all.size() % 4) scales_all.push_back(0);

    std::vector<float> xv((size_t) k);
    for (auto & v : xv) v = dist(rng);
    std::vector<int32_t> idv(n_used);
    for (int64_t i = 0; i < n_used; i++) idv[i] = (int32_t) i;

    const uint32_t n_tiles = (uint32_t) (k / tile_k);
    const size_t partials_n = (size_t) n_used * n_tiles * m;
    const size_t y_n = (size_t) n_used * m;

    uint2    * d_plane = (uint2 *) to_device(plane_all);
    uint32_t * d_scale;
    HIP_CHECK(hipMalloc(&d_scale, scales_all.size()));
    HIP_CHECK(hipMemcpy(d_scale, scales_all.data(), scales_all.size(), hipMemcpyHostToDevice));
    float    * d_x   = to_device(xv);
    int32_t  * d_ids = to_device(idv);
    float    * d_part, * d_y;
    HIP_CHECK(hipMalloc(&d_part, partials_n * 4));
    HIP_CHECK(hipMalloc(&d_y, y_n * 4));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    const uint32_t w_stride = (uint32_t) (k * m / 16);
    const uint32_t s_stride = (uint32_t) ((k / 32) * m / 4);
    const dim3 g1((uint32_t) ((m / 8 * 2 + 255) / 256), n_tiles, (uint32_t) n_used);
    const dim3 g2((uint32_t) ((m + 255) / 256), 1, (uint32_t) n_used);
    const dim3 b256(256);

    auto one_rep = [&]() {
        k_pass1<<<g1, b256, 0, stream>>>(d_plane, d_scale, d_x, d_part, d_ids,
            (uint32_t) k, (uint32_t) m, (uint32_t) tile_k, n_tiles, w_stride, s_stride, 0u);
        k_pass2<<<g2, b256, 0, stream>>>(d_part, d_y, (uint32_t) m, n_tiles);
    };

    one_rep();   // warm-up: first-use costs, excluded from the timed span
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));
    const auto t0 = std::chrono::steady_clock::now();
    HIP_CHECK(hipEventRecord(ev0, stream));
    for (int r = 0; r < reps; r++) one_rep();
    HIP_CHECK(hipEventRecord(ev1, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    const double wall_us = std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - t0).count() / reps;
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    const double gpu_us = (double) ms * 1000.0 / reps;

    std::vector<float> y(y_n);
    HIP_CHECK(hipMemcpy(y.data(), d_y, y_n * 4, hipMemcpyDeviceToHost));

    // Reference: dequant (exact) + double-accumulated matvec on the CPU.
    std::vector<double> ref(y_n);
    {
        std::vector<float> W((size_t) (k * m));
        for (int64_t e = 0; e < n_used; e++) {
            dequant(expert_blocks[e].data(), k, m, W.data());
            for (int64_t col = 0; col < m; col++) {
                double acc = 0.0;
                const float * row = W.data() + col * k;
                for (int64_t kk = 0; kk < k; kk++) acc += (double) row[kk] * xv[kk];
                ref[(size_t) (e * m + col)] = acc;
            }
        }
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    const double bytes = (double) n_used * m * (k / 32) * 17;
    printf("%s [dev %d]: 2-pass mxfp4, k=%lld m=%lld experts=%lld tile_k=%lld (%u tiles), %d reps\n",
           prop.name, dev, (long long) k, (long long) m, (long long) n_used,
           (long long) tile_k, n_tiles, reps);
    printf("  gpu  %10.1f us  %8.1f GB/s   (wall %.1f us)\n",
           gpu_us, bytes / (gpu_us * 1e-6) / 1e9, wall_us);
    print_kernel_info((const void *) k_pass1, "pass1");
    const bool ok = check(y, ref);

    HIP_CHECK(hipEventDestroy(ev0));
    HIP_CHECK(hipEventDestroy(ev1));
    HIP_CHECK(hipStreamDestroy(stream));
    for (void * pdel : { (void *) d_plane, (void *) d_scale, (void *) d_x,
                         (void *) d_ids, (void *) d_part, (void *) d_y }) hipFree(pdel);
    return ok ? 0 : 1;
}

// --- tp: the E9 block, one die's pipeline or all four dies for real ---------
//
// Each die-slice holds columns [I*d, I*(d+1)) of gate and up (fused into one
// m=2I matmul) and the matching k-rows of down, for every expert. Four
// kernels per die: pass1(GU) -> mid(reduce+clamp+SwiGLU) -> pass1(down) ->
// final(reduce x router weight). The host sums the four partial [M] outputs.
//
// single mode: all four slices sequential on one die, GPU-event timing —
//   comparable to the Vulkan probe's 113 us/die.
// multi mode: slice d on physical die d, one stream each, per-rep H2D of x
//   and D2H of the partials included, wall-clock timing — the ROCm analog of
//   the in-process per-layer cost (Vulkan probe round 212 us + 113 GPU;
//   moe-serv in-process 439 us).

struct tp_slice {
    int dev;
    hipStream_t stream;
    uint2    * gu_plane;  uint32_t * gu_scale;
    uint2    * dn_plane;  uint32_t * dn_scale;
    float    * x;         int32_t  * ids;      float * wt;
    float    * part_gu;   float    * h;
    float    * part_dn;   float    * y;
};

static int run_tp(int reps, int64_t tile_k, bool multi, int dev) {
    const int64_t K = 4096, INTER = 2048, MOUT = 4096, SLOTS = 6, DIES = 4;
    const int64_t ISL = INTER / DIES;                 // 512: die slice of intermediate
    const int64_t M2  = 2 * ISL;                      // fused gate|up columns
    const float gmin = -1e30f, gmax = 10.0f;          // representative deepseek4 clamps
    const float umin = -10.0f,  umax = 10.0f;

    const uint32_t gu_tiles = (uint32_t) (K / tile_k);
    const uint32_t dn_tiles = (uint32_t) (ISL / tile_k);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::vector<uint8_t>> gate_b(SLOTS), up_b(SLOTS), down_b(SLOTS);
    for (int64_t e = 0; e < SLOTS; e++) {
        gate_b[e] = random_blocks(rng, K, INTER);
        up_b[e]   = random_blocks(rng, K, INTER);
        down_b[e] = random_blocks(rng, INTER, MOUT);
    }
    std::vector<float> xv((size_t) K), wv((size_t) SLOTS);
    for (auto & v : xv) v = dist(rng);
    for (auto & v : wv) v = 0.1f + 0.15f * dist(rng);
    std::vector<int32_t> idv(SLOTS);
    for (int64_t i = 0; i < SLOTS; i++) idv[i] = (int32_t) i;

    // Reference: the whole unsliced block per slot, f32 dequant + double accum.
    std::vector<double> ref((size_t) (SLOTS * MOUT), 0.0);
    {
        std::vector<float> G((size_t) (K * INTER)), U((size_t) (K * INTER)), D((size_t) (INTER * MOUT));
        std::vector<double> h((size_t) INTER);
        for (int64_t e = 0; e < SLOTS; e++) {
            dequant(gate_b[e].data(), K, INTER, G.data());
            dequant(up_b[e].data(),   K, INTER, U.data());
            dequant(down_b[e].data(), INTER, MOUT, D.data());
            for (int64_t c = 0; c < INTER; c++) {
                double g = 0.0, u = 0.0;
                for (int64_t kk = 0; kk < K; kk++) {
                    g += (double) G[c * K + kk] * xv[kk];
                    u += (double) U[c * K + kk] * xv[kk];
                }
                g = std::min(std::max(g, (double) gmin), (double) gmax);
                u = std::min(std::max(u, (double) umin), (double) umax);
                h[c] = (g / (1.0 + exp(-g))) * u;
            }
            for (int64_t c = 0; c < MOUT; c++) {
                double acc = 0.0;
                for (int64_t kk = 0; kk < INTER; kk++) acc += (double) D[c * INTER + kk] * h[kk];
                ref[(size_t) (e * MOUT + c)] = acc * wv[e];
            }
        }
    }

    // Per-die repacked slices.
    std::vector<tp_slice> S(DIES);
    std::vector<std::vector<uint32_t>> gu_plane_h(DIES), dn_plane_h(DIES);
    std::vector<std::vector<uint8_t>>  gu_scale_h(DIES), dn_scale_h(DIES);
    for (int64_t d = 0; d < DIES; d++) {
        std::vector<uint32_t> plane((size_t) (K * M2 / 8));
        std::vector<uint8_t>  scales((size_t) ((K / 32) * M2));
        std::vector<uint32_t> plane2((size_t) (ISL * MOUT / 8));
        std::vector<uint8_t>  scales2((size_t) ((ISL / 32) * MOUT));
        for (int64_t e = 0; e < SLOTS; e++) {
            std::fill(plane.begin(), plane.end(), 0);
            std::fill(scales.begin(), scales.end(), 0);
            repack_slice(gate_b[e].data(), K, ISL * d, ISL, M2, 0,   plane, scales);
            repack_slice(up_b[e].data(),   K, ISL * d, ISL, M2, ISL, plane, scales);
            gu_plane_h[d].insert(gu_plane_h[d].end(), plane.begin(), plane.end());
            gu_scale_h[d].insert(gu_scale_h[d].end(), scales.begin(), scales.end());

            std::fill(plane2.begin(), plane2.end(), 0);
            std::fill(scales2.begin(), scales2.end(), 0);
            repack_kslice(down_b[e].data(), INTER, MOUT, ISL * d / 32, ISL / 32, plane2, scales2);
            dn_plane_h[d].insert(dn_plane_h[d].end(), plane2.begin(), plane2.end());
            dn_scale_h[d].insert(dn_scale_h[d].end(), scales2.begin(), scales2.end());
        }
    }

    for (int64_t d = 0; d < DIES; d++) {
        tp_slice & s = S[d];
        s.dev = multi ? (int) d : dev;
        HIP_CHECK(hipSetDevice(s.dev));
        HIP_CHECK(hipStreamCreate(&s.stream));
        s.gu_plane = (uint2 *) to_device(gu_plane_h[d]);
        s.gu_scale = (uint32_t *) (void *) to_device(gu_scale_h[d]);
        s.dn_plane = (uint2 *) to_device(dn_plane_h[d]);
        s.dn_scale = (uint32_t *) (void *) to_device(dn_scale_h[d]);
        s.x   = to_device(xv);
        s.ids = to_device(idv);
        s.wt  = to_device(wv);
        HIP_CHECK(hipMalloc(&s.part_gu, (size_t) SLOTS * gu_tiles * M2 * 4));
        HIP_CHECK(hipMalloc(&s.h,       (size_t) SLOTS * ISL * 4));
        HIP_CHECK(hipMalloc(&s.part_dn, (size_t) SLOTS * dn_tiles * MOUT * 4));
        HIP_CHECK(hipMalloc(&s.y,       (size_t) SLOTS * MOUT * 4));
    }
    // In single mode all slices share one device and one stream keeps them
    // sequential, like the Vulkan probe. Use slice 0's stream for all.
    if (!multi) for (int64_t d = 1; d < DIES; d++) S[d].stream = S[0].stream;

    const uint32_t gu_wstride = (uint32_t) (K * M2 / 16);
    const uint32_t gu_sstride = (uint32_t) ((K / 32) * M2 / 4);
    const uint32_t dn_wstride = (uint32_t) (ISL * MOUT / 16);
    const uint32_t dn_sstride = (uint32_t) ((ISL / 32) * MOUT / 4);
    const dim3 b256(256);
    const dim3 g_gu((uint32_t) ((M2 / 8 * 2 + 255) / 256), gu_tiles, (uint32_t) SLOTS);
    const dim3 g_mid((uint32_t) ((ISL + 255) / 256), 1, (uint32_t) SLOTS);
    const dim3 g_dn((uint32_t) ((MOUT / 8 * 2 + 255) / 256), dn_tiles, (uint32_t) SLOTS);
    const dim3 g_fin((uint32_t) ((MOUT + 255) / 256), 1, (uint32_t) SLOTS);

    auto die_pipeline = [&](tp_slice & s) {
        k_pass1<<<g_gu, b256, 0, s.stream>>>(s.gu_plane, s.gu_scale, s.x, s.part_gu, s.ids,
            (uint32_t) K, (uint32_t) M2, (uint32_t) tile_k, gu_tiles, gu_wstride, gu_sstride, 0u);
        k_tp_mid<<<g_mid, b256, 0, s.stream>>>(s.part_gu, s.h,
            (uint32_t) ISL, gu_tiles, gmin, gmax, umin, umax);
        k_pass1<<<g_dn, b256, 0, s.stream>>>(s.dn_plane, s.dn_scale, s.h, s.part_dn, s.ids,
            (uint32_t) ISL, (uint32_t) MOUT, (uint32_t) tile_k, dn_tiles, dn_wstride, dn_sstride,
            (uint32_t) ISL);
        k_tp_final<<<g_fin, b256, 0, s.stream>>>(s.part_dn, s.wt, s.y, (uint32_t) MOUT, dn_tiles);
    };

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, S[0].dev));

    if (!multi) {
        HIP_CHECK(hipSetDevice(S[0].dev));
        for (auto & s : S) die_pipeline(s);   // warm-up
        HIP_CHECK(hipStreamSynchronize(S[0].stream));

        hipEvent_t ev0, ev1;
        HIP_CHECK(hipEventCreate(&ev0));
        HIP_CHECK(hipEventCreate(&ev1));
        HIP_CHECK(hipEventRecord(ev0, S[0].stream));
        for (int r = 0; r < reps; r++) for (auto & s : S) die_pipeline(s);
        HIP_CHECK(hipEventRecord(ev1, S[0].stream));
        HIP_CHECK(hipStreamSynchronize(S[0].stream));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        const double per_die_us = (double) ms * 1000.0 / reps / DIES;
        printf("%s [dev %d]: tp block single-die, tile_k=%lld, %d reps\n",
               prop.name, S[0].dev, (long long) tile_k, reps);
        printf("  gpu  %10.1f us/die  (4 kernels: GU %u tiles, mid, down %u tiles, final)\n",
               per_die_us, gu_tiles, dn_tiles);
        HIP_CHECK(hipEventDestroy(ev0));
        HIP_CHECK(hipEventDestroy(ev1));
    } else {
        float * hx = nullptr;      // pinned staging for per-rep transfers
        float * hy = nullptr;
        HIP_CHECK(hipHostMalloc(&hx, (size_t) K * 4));
        HIP_CHECK(hipHostMalloc(&hy, (size_t) DIES * SLOTS * MOUT * 4));
        memcpy(hx, xv.data(), (size_t) K * 4);

        std::vector<float> ysum((size_t) SLOTS * MOUT);
        auto one_rep = [&]() {
            for (auto & s : S) {
                HIP_CHECK(hipSetDevice(s.dev));
                HIP_CHECK(hipMemcpyAsync(s.x, hx, (size_t) K * 4, hipMemcpyHostToDevice, s.stream));
                die_pipeline(s);
                HIP_CHECK(hipMemcpyAsync(hy + (&s - S.data()) * SLOTS * MOUT, s.y,
                    (size_t) SLOTS * MOUT * 4, hipMemcpyDeviceToHost, s.stream));
            }
            for (auto & s : S) {
                HIP_CHECK(hipSetDevice(s.dev));
                while (hipStreamQuery(s.stream) == hipErrorNotReady) {}   // poll, like moe_tp
            }
            memcpy(ysum.data(), hy, ysum.size() * 4);
            for (int64_t d = 1; d < DIES; d++) {
                const float * yp = hy + d * SLOTS * MOUT;
                for (size_t i = 0; i < ysum.size(); i++) ysum[i] += yp[i];
            }
        };

        one_rep();   // warm-up
        std::vector<double> times;
        for (int r = 0; r < reps; r++) {
            const auto t0 = std::chrono::steady_clock::now();
            one_rep();
            times.push_back(std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(times.begin(), times.end());
        printf("%s x%lld dies: tp block multi-die (H2D x + 4 kernels/die + D2H + host sum), %d reps\n",
               prop.name, (long long) DIES, reps);
        printf("  wall/layer  median %.1f us  p10 %.1f  p90 %.1f\n",
               times[times.size() / 2], times[times.size() / 10], times[times.size() * 9 / 10]);
        HIP_CHECK(hipHostFree(hx));
        HIP_CHECK(hipHostFree(hy));
    }

    // Correctness either way: read every slice's partials and sum on the host.
    std::vector<float> ysum((size_t) SLOTS * MOUT, 0.0f);
    std::vector<float> part((size_t) SLOTS * MOUT);
    for (auto & s : S) {
        HIP_CHECK(hipSetDevice(s.dev));
        HIP_CHECK(hipMemcpy(part.data(), s.y, part.size() * 4, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < ysum.size(); i++) ysum[i] += part[i];
    }
    const bool ok = check(ysum, ref);

    for (int64_t d = 0; d < DIES; d++) {
        tp_slice & s = S[d];
        HIP_CHECK(hipSetDevice(s.dev));
        if (multi || d == 0) HIP_CHECK(hipStreamDestroy(s.stream));
        for (void * pdel : { (void *) s.gu_plane, (void *) s.gu_scale, (void *) s.dn_plane,
                             (void *) s.dn_scale, (void *) s.x, (void *) s.ids, (void *) s.wt,
                             (void *) s.part_gu, (void *) s.h, (void *) s.part_dn, (void *) s.y })
            hipFree(pdel);
    }
    return ok ? 0 : 1;
}

// --- latency: null-kernel floors, the vk-latency analog ---------------------

static int run_latency(int iters) {
    int n_dev = 0;
    HIP_CHECK(hipGetDeviceCount(&n_dev));
    printf("latency, %d iters, %d device(s)   (Vulkan floors: submit 9 us, "
           "round polled ~59-67, 4-die round 87)\n", iters, n_dev);

    std::vector<hipStream_t> streams(n_dev);
    for (int d = 0; d < n_dev; d++) {
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipStreamCreate(&streams[d]));
        k_null<<<dim3(1), dim3(64), 0, streams[d]>>>();   // warm-up: module load
        HIP_CHECK(hipStreamSynchronize(streams[d]));
    }

    // Rung A: sustained launch cost — back-to-back async launches, one sync.
    {
        HIP_CHECK(hipSetDevice(0));
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) k_null<<<dim3(1), dim3(64), 0, streams[0]>>>();
        const double launch_us = std::chrono::duration<double, std::micro>(
                                     std::chrono::steady_clock::now() - t0).count() / iters;
        HIP_CHECK(hipStreamSynchronize(streams[0]));
        printf("  launch (async, sustained)          %8.2f us\n", launch_us);
    }

    // Rung B: single launch -> host observes completion, polled and blocking.
    {
        HIP_CHECK(hipSetDevice(0));
        std::vector<double> poll, block;
        for (int i = 0; i < iters; i++) {
            auto t0 = std::chrono::steady_clock::now();
            k_null<<<dim3(1), dim3(64), 0, streams[0]>>>();
            while (hipStreamQuery(streams[0]) == hipErrorNotReady) {}
            poll.push_back(std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - t0).count());

            t0 = std::chrono::steady_clock::now();
            k_null<<<dim3(1), dim3(64), 0, streams[0]>>>();
            HIP_CHECK(hipStreamSynchronize(streams[0]));
            block.push_back(std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t0).count());
        }
        printf("  round trip, polled                 %8.2f us  (median)\n", median(poll));
        printf("  round trip, blocking sync          %8.2f us  (median)\n", median(block));
    }

    // Rung C: all dies — launch on every device, then observe all done.
    if (n_dev > 1) {
        std::vector<double> round;
        for (int i = 0; i < iters; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int d = 0; d < n_dev; d++) {
                HIP_CHECK(hipSetDevice(d));
                k_null<<<dim3(1), dim3(64), 0, streams[d]>>>();
            }
            for (int d = 0; d < n_dev; d++) {
                HIP_CHECK(hipSetDevice(d));
                while (hipStreamQuery(streams[d]) == hipErrorNotReady) {}
            }
            round.push_back(std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t0).count());
        }
        printf("  %d-die round, launch-all/poll-all   %8.2f us  (median)\n", n_dev, median(round));
    }

    for (int d = 0; d < n_dev; d++) {
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipStreamDestroy(streams[d]));
    }
    return 0;
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    int64_t k = 4096, m = 2048, experts = 6, tile_k = 128;
    int reps = 100, iters = 1000, dev = 0;
    std::string mode = argc > 1 ? argv[1] : "";

    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> int64_t { return i + 1 < argc ? atoll(argv[++i]) : 0; };
        if      (a == "--k")       k = next();
        else if (a == "--m")       m = next();
        else if (a == "--experts") experts = next();
        else if (a == "--tile")    tile_k = next();
        else if (a == "--reps")    reps = (int) next();
        else if (a == "--iters")   iters = (int) next();
        else if (a == "--device")  dev = (int) next();
        else { fprintf(stderr, "unknown flag %s\n", a.c_str()); return 2; }
    }

    if (mode == "matmul") return run_matmul(k, m, experts, reps, tile_k, dev);
    if (mode == "tp")     return run_tp(reps, tile_k, false, dev);
    if (mode == "tp4")    return run_tp(reps, tile_k, true, dev);
    if (mode == "latency") return run_latency(iters);
    if (mode == "all") {
        int rc = 0;
        rc |= run_matmul(4096, 2048, 6, reps, tile_k, dev);   // gate/up
        rc |= run_matmul(2048, 4096, 6, reps, tile_k, dev);   // down
        rc |= run_tp(reps, tile_k, false, dev);
        rc |= run_tp(reps, tile_k, true, dev);
        rc |= run_latency(iters);
        return rc;
    }

    fprintf(stderr,
        "usage: %s <mode> [flags]\n"
        "  matmul   2-pass mxfp4 vecmat   --k 4096 --m 2048 --experts 6 --tile 128 --reps 100 --device 0\n"
        "  tp       TP block, one die's pipeline (GPU time, vs Vulkan 113 us/die)\n"
        "  tp4      TP block on all four dies, transfers + host sum included (wall)\n"
        "  latency  null-kernel launch/sync floors   --iters 1000\n"
        "  all      the full comparison table\n", argv[0]);
    return 2;
}
