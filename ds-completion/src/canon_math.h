// Elementwise math with the same bits on the CPU (g++) and the GPU (hipcc): the expert
// activation (swiglu) of the placement-invariant expert numerics (canon_dot.h).
//
// The libraries' expf differ in the last bit, and hipcc fuses a * b + c into an fma where g++
// does not; so: only plain operations and explicit fmaf, correctly rounded division on the GPU
// (__fdiv_rn), and nothing a compiler could contract.
#pragma once

#include <cmath>

#if defined(__HIPCC__)
#define CANON_HD __host__ __device__
#else
#define CANON_HD
#endif

CANON_HD inline float canon_div(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}

// e^x within ~1 ulp (Cephes expf: n = round(x log2 e), r = x - n ln 2 in two parts, a degree-6
// polynomial, then 2^n)
CANON_HD inline float canon_expf(float x) {
  if (x > 88.72283f) return INFINITY;
  if (x < -103.97208f) return 0.0f;
  const float n = rintf(x * 1.44269504088896341f);
  float r = fmaf(n, -0.693359375f, x);
  r = fmaf(n, 2.12194440e-4f, r);
  const float z = r * r;
  float p = 1.9875691500e-4f;
  p = fmaf(p, r, 1.3981999507e-3f);
  p = fmaf(p, r, 8.3334519073e-3f);
  p = fmaf(p, r, 4.1665795894e-2f);
  p = fmaf(p, r, 1.6666665459e-1f);
  p = fmaf(p, r, 5.0000001201e-1f);
  const float y = fmaf(p, z, r) + 1.0f;
  return ldexpf(y, (int)n);
}

// silu(g) = g / (1 + e^-g)
CANON_HD inline float canon_silu(float g) { return canon_div(g, 1.0f + canon_expf(-g)); }
