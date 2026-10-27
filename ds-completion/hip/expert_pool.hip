#include "expert_pool.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    hipError_t e_ = (x);                                                                           \
    if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); \
  } while (0)

static size_t align256(size_t n) { return (n + 255) & ~(size_t)255; }

// one matrix of N rows x K: codes N * K / 2 bytes, scales N * K / 32 bytes
static size_t codes_bytes(int N, int K) { return (size_t)N * K / 2; }
static size_t scales_bytes(int N, int K) { return (size_t)N * K / 32; }

size_t pool_slot_bytes(int E, int F) {
  // gate and up: N = F rows of K = E; down: N = E rows of K = F (the same sizes)
  return 3 * (align256(codes_bytes(F, E)) + align256(scales_bytes(F, E)));
}

ExpertPool pool_create(int dev, int n_slots, int E, int F) {
  int prev;
  CHECK(hipGetDevice(&prev));
  CHECK(hipSetDevice(dev));
  ExpertPool p;
  p.dev = dev;
  p.n_slots = n_slots;
  p.slot_bytes = pool_slot_bytes(E, F);
  if (n_slots > 0) CHECK(hipMalloc(&p.base, (size_t)n_slots * p.slot_bytes));
  CHECK(hipMalloc(&p.stage, (size_t)F * E / 32 * 17));
  // slot layout: gate codes, gate scales, up codes, up scales, down codes, down scales
  size_t off = 0;
  auto mat = [&](int N, int K) {
    GpuMatrix m;
    m.N = N;
    m.K = K;
    m.codes = p.base + off;
    off += align256(codes_bytes(N, K));
    m.scales = p.base + off;
    off += align256(scales_bytes(N, K));
    m.codes_stride = m.scales_stride = p.slot_bytes;
    return m;
  };
  p.view.gate = mat(F, E);
  p.view.up = mat(F, E);
  p.view.down = mat(E, F);
  CHECK(hipSetDevice(prev));
  return p;
}

int pool_slots_that_fit(int dev, int E, int F, size_t reserve) {
  int prev;
  CHECK(hipGetDevice(&prev));
  CHECK(hipSetDevice(dev));
  size_t free_b, total_b;
  CHECK(hipMemGetInfo(&free_b, &total_b));
  CHECK(hipSetDevice(prev));
  const size_t stage = (size_t)F * E / 32 * 17;
  if (free_b < reserve + stage) return 0;
  return (int)((free_b - reserve - stage) / pool_slot_bytes(E, F));
}

void pool_load(ExpertPool& p, int slot, const void* gate, const void* up, const void* down, hipStream_t s) {
  if (slot < 0 || slot >= p.n_slots) throw std::runtime_error("pool_load: bad slot");
  const size_t base = (size_t)slot * p.slot_bytes;
  for (auto [src, m] : {std::make_pair(gate, &p.view.gate), std::make_pair(up, &p.view.up),
                        std::make_pair(down, &p.view.down)}) {
    const size_t n_blocks = (size_t)m->N * m->K / 32;
    CHECK(hipMemcpyAsync(p.stage, src, n_blocks * 17, hipMemcpyHostToDevice, s));
    fp4_unpack(p.stage, n_blocks, m->codes + base, m->scales + base, s);
  }
}

void pool_free(ExpertPool& p) {
  if (p.dev < 0) return;
  int prev;
  CHECK(hipGetDevice(&prev));
  CHECK(hipSetDevice(p.dev));
  (void)hipFree(p.base);
  (void)hipFree(p.stage);
  CHECK(hipSetDevice(prev));
  p = ExpertPool{};
}

std::vector<std::pair<int, int>> read_placement(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read placement " + path);
  std::string line;
  std::getline(f, line);
  if (line.rfind("# ds-completion expert placement v1", 0) != 0) throw std::runtime_error(path + ": not a placement v1 file");
  std::vector<std::pair<int, int>> r;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    int L, e;
    if (!(is >> L >> e)) throw std::runtime_error(path + ": bad line: " + line);
    r.emplace_back(L, e);
  }
  return r;
}

Placement place_experts(const std::vector<std::pair<int, int>>& ranked, int n_layer, int n_expert,
                        const std::vector<int>& capacity, const std::vector<bool>& layers_loaded) {
  const int n_dev = (int)capacity.size();
  Placement p;
  p.n_expert = n_expert;
  p.die.assign((size_t)n_layer * n_expert, -1);
  p.slot.assign((size_t)n_layer * n_expert, -1);
  p.used.assign(n_dev, 0);
  std::vector<int> next(n_layer, 0);  // per layer: the die to try first
  int free_total = 0;
  for (int c : capacity) free_total += c;
  for (auto [L, e] : ranked) {
    if (free_total == 0) break;
    if (L < 0 || L >= n_layer || e < 0 || e >= n_expert) throw std::runtime_error("placement: pair out of range");
    if (!layers_loaded[L] || p.die[(size_t)L * n_expert + e] >= 0) continue;
    for (int t = 0; t < n_dev; t++) {
      const int d = (next[L] + t) % n_dev;
      if (p.used[d] >= capacity[d]) continue;
      p.die[(size_t)L * n_expert + e] = (int16_t)d;
      p.slot[(size_t)L * n_expert + e] = (int16_t)p.used[d]++;
      free_total--;
      next[L] = (d + 1) % n_dev;
      break;
    }
  }
  return p;
}
