// test-topk: the indexer's selection kernels on given scores, for ref/check_topk.py (which
// compares them with the reference's torch.topk and select_candidate_blocks).
//
//   test-topk in.bin out.bin
//
// in: per case int32 mode, n, k, block, topk_blocks, then float scores[n]
//   mode 0: k_topk(scores, n, k)                                  -> int32 idx[k]
//   mode 1: k_candidate_blocks(scores, n, block, topk_blocks)     -> uint8 keep[ceil(n / block)]
//   mode 2: candidates as 1, then k_mask_candidates + k_topk(k)   -> int32 idx[k]
#include "kernels.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

int main(int argc, char** argv) {
  if (argc != 3) { fprintf(stderr, "usage: test-topk in.bin out.bin\n"); return 2; }
  FILE* in = fopen(argv[1], "rb");
  FILE* out = fopen(argv[2], "wb");
  if (!in || !out) { fprintf(stderr, "cannot open files\n"); return 2; }
  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  int hdr[5];
  int cases = 0;
  while (fread(hdr, 4, 5, in) == 5) {
    const int mode = hdr[0], n = hdr[1], k = hdr[2], block = hdr[3], tb = hdr[4];
    std::vector<float> sc(n);
    if (fread(sc.data(), 4, n, in) != (size_t)n) { fprintf(stderr, "short read\n"); return 1; }
    float *d_sc, *d_bs;
    int *d_idx, *d_bidx;
    uint8_t* d_keep;
    const int nb = (n + block - 1) / (block ? block : 1);
    CHECK(hipMalloc(&d_sc, (size_t)n * 4));
    CHECK(hipMalloc(&d_idx, (size_t)(k > 0 ? k : 1) * 4));
    CHECK(hipMalloc(&d_bs, (size_t)(nb + 1) * 4));
    CHECK(hipMalloc(&d_bidx, (size_t)(tb > 0 ? tb : 1) * 4));
    CHECK(hipMalloc(&d_keep, (size_t)nb + 1));
    CHECK(hipMemcpy(d_sc, sc.data(), (size_t)n * 4, hipMemcpyHostToDevice));
    if (mode == 1 || mode == 2) k_candidate_blocks(d_sc, n, block, tb, d_bs, d_bidx, d_keep, s);
    if (mode == 2) k_mask_candidates(d_sc, n, block, d_keep, s);
    if (mode == 0 || mode == 2) k_topk(d_sc, n, k, d_idx, s);
    CHECK(hipStreamSynchronize(s));
    if (mode == 1) {
      std::vector<uint8_t> keep(nb);
      CHECK(hipMemcpy(keep.data(), d_keep, nb, hipMemcpyDeviceToHost));
      fwrite(keep.data(), 1, nb, out);
    } else {
      std::vector<int> idx(k);
      CHECK(hipMemcpy(idx.data(), d_idx, (size_t)k * 4, hipMemcpyDeviceToHost));
      fwrite(idx.data(), 4, k, out);
    }
    for (void* p : {(void*)d_sc, (void*)d_idx, (void*)d_bs, (void*)d_bidx, (void*)d_keep}) CHECK(hipFree(p));
    cases++;
  }
  fclose(out);
  fprintf(stderr, "test-topk: %d cases\n", cases);
  return 0;
}
