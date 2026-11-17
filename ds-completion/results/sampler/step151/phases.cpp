#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
extern "C" __m512d _ZGVeN8v_exp(__m512d);
using C = std::chrono::steady_clock;
static double ms(C::time_point a) { return std::chrono::duration<double, std::milli>(C::now() - a).count(); }
int main() {
  const int n = 129280, R = 300;
  std::mt19937 g(1); std::normal_distribution<float> d(0, 2);
  std::vector<float> l(n); for (float& v : l) v = d(g); for (int j = 0; j < 10; j++) l[g() % n] = 18 + j;
  std::vector<double> p(n); double sink = 0;
  auto t = C::now(); for (int r = 0; r < R; r++) sink += std::max_element(l.begin(), l.end()) - l.begin(); printf("max_element %.3f ms\n", ms(t) / R);
  t = C::now(); for (int r = 0; r < R; r++) { float m = l[0]; for (int i = 1; i < n; i++) m = std::max(m, l[i]); sink += m; l[r] += 1e-3f; } printf("max value (vectorizable) %.3f ms\n", ms(t) / R);
  const double top = *std::max_element(l.begin(), l.end());
  t = C::now(); for (int r = 0; r < R; r++) { for (int i = 0; i < n; i++) p[i] = std::exp((l[i] - top) * 1.0); sink += p[r]; } printf("scalar exp %.3f ms\n", ms(t) / R);
  t = C::now(); for (int r = 0; r < R; r++) { const __m512d vt = _mm512_set1_pd(top); for (int i = 0; i < n; i += 8) _mm512_storeu_pd(&p[i], _ZGVeN8v_exp(_mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(&l[i])), vt))); sink += p[r]; } printf("vector exp %.3f ms\n", ms(t) / R);
  t = C::now(); for (int r = 0; r < R; r++) { double s = 0; for (int i = 0; i < n; i++) s += p[i]; sink += s; p[r] *= 1.0000001; } printf("ordered sum %.3f ms\n", ms(t) / R);
  t = C::now(); for (int r = 0; r < R; r++) { std::vector<int> c; for (int i = 0; i < n; i++) if (p[i] >= 1e-7) c.push_back(i); sink += c.size(); p[r] *= 1.0000001; } printf("candidates %.3f ms\n", ms(t) / R);
  t = C::now(); for (int r = 0; r < R; r++) { std::vector<double> q(n); sink += q[r]; } printf("alloc+zero 1 MB %.3f ms\n", ms(t) / R);
  printf("%g\n", sink);
}
