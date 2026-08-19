// ceilings — the machine's memory and launch floors, measured separately from
// any model (the hip-moe lesson: without these denominators, every "GB/s
// effective" claim attributes the shortfall to whatever mechanism was in mind).
//
// Three probes on M2 Ultra:
//   cpu   — multithreaded sequential f32 sum over a large buffer, thread sweep.
//           QoS USER_INTERACTIVE so threads prefer P cores (macOS has no
//           affinity API; the scheduler decides).
//   gpu   — Metal kernel, grid-strided float4 sum over a large shared-mode
//           buffer, timed with the command buffer's own GPU timestamps.
//   floor — null-kernel launch costs: dispatch-to-dispatch inside one command
//           buffer (GPU time / N), and submit -> host-observes-done round
//           trips (the number that bounds any GPU+CPU overlap design).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <pthread.h>

static const char * MSL = R"(
#include <metal_stdlib>
using namespace metal;

kernel void knull() {}

kernel void ksum(device const float4 * in  [[buffer(0)]],
                 device float        * out [[buffer(1)]],
                 constant uint       & n4  [[buffer(2)]],
                 uint gid [[thread_position_in_grid]],
                 uint gsz [[threads_per_grid]]) {
    float4 acc = 0.0f;
    for (uint i = gid; i < n4; i += gsz) acc += in[i];
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}
)";

static double now_us() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// --- CPU streaming ----------------------------------------------------------
static void cpu_probe(size_t gib, const std::vector<int> & tsweep, int reps) {
    const size_t n = gib * (1ull << 30) / 4;
    std::vector<float> buf(n, 1.0f);
    printf("cpu streaming, %zu GiB buffer, best of %d passes\n", gib, reps);
    printf("%-3s %10s\n", "t", "GB/s");
    for (int t : tsweep) {
        double best = 0;
        for (int r = 0; r < reps; r++) {
            std::vector<std::thread> th;
            std::vector<double> sums(t);
            const double t0 = now_us();
            for (int i = 0; i < t; i++) {
                th.emplace_back([&, i] {
                    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
                    const size_t lo = n / t * i, hi = i == t - 1 ? n : n / t * (i + 1);
                    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                    const float * p = buf.data();
                    size_t j = lo;
                    for (; j + 16 <= hi; j += 16) {
                        a0 += p[j+0] + p[j+4] + p[j+ 8] + p[j+12];
                        a1 += p[j+1] + p[j+5] + p[j+ 9] + p[j+13];
                        a2 += p[j+2] + p[j+6] + p[j+10] + p[j+14];
                        a3 += p[j+3] + p[j+7] + p[j+11] + p[j+15];
                    }
                    for (; j < hi; j++) a0 += p[j];
                    sums[i] = a0 + a1 + a2 + a3;
                });
            }
            for (auto & x : th) x.join();
            const double us = now_us() - t0;
            double total = 0;
            for (double s : sums) total += s;
            if (total < 0) printf("?");   // keep the sum observable
            best = fmax(best, n * 4.0 / us / 1e3);
        }
        printf("%-3d %10.1f\n", t, best);
    }
}

// --- GPU probes ---------------------------------------------------------------
int main(int argc, char ** argv) {
    bool do_cpu = true, do_gpu = true, do_floor = true;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--cpu"))   { do_gpu = do_floor = false; }
        else if (!strcmp(argv[i], "--gpu"))   { do_cpu = do_floor = false; }
        else if (!strcmp(argv[i], "--floor")) { do_cpu = do_gpu = false; }
        else { fprintf(stderr, "usage: %s [--cpu|--gpu|--floor]\n", argv[0]); return 2; }
    }

    if (do_cpu) {
        cpu_probe(8, {4, 8, 12, 16, 20, 24, 32}, 5);
        printf("\n");
    }
    if (!do_gpu && !do_floor) return 0;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }
        printf("gpu: %s\n", dev.name.UTF8String);

        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:MSL]
                                               options:nil error:&err];
        if (!lib) { fprintf(stderr, "MSL: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLComputePipelineState> pnull =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"knull"] error:&err];
        id<MTLComputePipelineState> psum =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ksum"] error:&err];
        id<MTLCommandQueue> q = [dev newCommandQueue];

        if (do_gpu) {
            const size_t gib = 8;
            const size_t bytes = gib << 30;
            const uint32_t n4 = (uint32_t) (bytes / 16);
            const uint32_t nthreads = 1u << 18;
            id<MTLBuffer> in  = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> out = [dev newBufferWithLength:nthreads * 4 options:MTLResourceStorageModeShared];
            memset(in.contents, 1, bytes);   // touch every page before timing
            printf("gpu streaming, %zu GiB buffer, %u threads, 5 passes\n", gib, nthreads);
            for (int r = 0; r < 5; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:psum];
                [enc setBuffer:in offset:0 atIndex:0];
                [enc setBuffer:out offset:0 atIndex:1];
                [enc setBytes:&n4 length:4 atIndex:2];
                [enc dispatchThreads:MTLSizeMake(nthreads, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const double s = cb.GPUEndTime - cb.GPUStartTime;
                printf("  pass %d: %.1f GB/s (%.2f ms GPU)\n", r, bytes / s / 1e9, s * 1e3);
            }
            printf("\n");
        }

        if (do_floor) {
            // (a) dispatch-to-dispatch inside one command buffer
            {
                const int N = 1000;
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pnull];
                for (int i = 0; i < N; i++)
                    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                printf("floor: null dispatch, same encoder, x%d: %.2f us each (GPU time)\n",
                       N, (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / N);
            }
            // (b) submit -> host observes done, one null kernel per command buffer
            {
                const int N = 200;
                // warmup
                for (int i = 0; i < 10; i++) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pnull];
                    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    [enc endEncoding];
                    [cb commit]; [cb waitUntilCompleted];
                }
                const double t0 = now_us();
                for (int i = 0; i < N; i++) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pnull];
                    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    [enc endEncoding];
                    [cb commit]; [cb waitUntilCompleted];
                }
                printf("floor: encode+commit+wait round trip, x%d: %.1f us each (wall)\n",
                       N, (now_us() - t0) / N);
            }
            // (c) sustained commit rate, wait only on the last
            {
                const int N = 1000;
                const double t0 = now_us();
                id<MTLCommandBuffer> last = nil;
                for (int i = 0; i < N; i++) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pnull];
                    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    [enc endEncoding];
                    [cb commit];
                    last = cb;
                }
                [last waitUntilCompleted];
                printf("floor: sustained encode+commit, wait last, x%d: %.2f us each (wall)\n",
                       N, (now_us() - t0) / N);
            }
        }
    }
    return 0;
}
