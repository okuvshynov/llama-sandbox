#include "expert_loader.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    hipError_t e_ = (x);                                                                           \
    if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); \
  } while (0)

LoadStage load_stage_create(int dev, size_t bytes, int priority) {
  LoadStage st;
  st.dev = dev;
  CHECK(hipSetDevice(dev));
  CHECK(hipStreamCreateWithPriority(&st.s, hipStreamNonBlocking, priority));
  for (int b = 0; b < 2; b++) {
    CHECK(hipHostMalloc(&st.host[b], bytes, 0));
    CHECK(hipMalloc(&st.dstage[b], bytes));
    // blocking: the loader thread sleeps while it waits (a spinning wait costs the CPU expert
    // kernel ~5%)
    CHECK(hipEventCreateWithFlags(&st.free[b], hipEventDisableTiming | hipEventBlockingSync));
    CHECK(hipEventRecord(st.free[b], st.s));
  }
  return st;
}

void load_stage_free(LoadStage& st) {
  if (st.dev < 0) return;
  (void)hipSetDevice(st.dev);
  (void)hipStreamSynchronize(st.s);
  for (int b = 0; b < 2; b++) {
    (void)hipHostFree(st.host[b]);
    (void)hipFree(st.dstage[b]);
    (void)hipEventDestroy(st.free[b]);
  }
  (void)hipStreamDestroy(st.s);
  st.dev = -1;
}

void load_expert(LoadStage& st, ExpertPool& p, int slot, const Layer& l, int e, bool direct) {
  if (slot < 0 || slot >= p.n_slots) throw std::runtime_error("load_expert: bad slot");
  const size_t base = (size_t)slot * p.slot_bytes;
  for (auto [t, mat] : {std::make_pair(l.gate_exps, &p.view.gate), std::make_pair(l.up_exps, &p.view.up),
                        std::make_pair(l.down_exps, &p.view.down)}) {
    const size_t n_blocks = (size_t)mat->N * mat->K / 32, bytes = n_blocks * 17;
    const uint8_t* src = (const uint8_t*)t->data + (size_t)e * t->nb[2];
    const int b = st.next;
    st.next ^= 1;
    CHECK(hipEventSynchronize(st.free[b]));  // at most 2 matrices in flight
    if (!direct) {
      memcpy(st.host[b], src, bytes);
      src = st.host[b];
    }
    CHECK(hipMemcpyAsync(st.dstage[b], src, bytes, hipMemcpyHostToDevice, st.s));
    fp4_unpack(st.dstage[b], n_blocks, mat->codes + base, mat->scales + base, st.s);
    CHECK(hipEventRecord(st.free[b], st.s));
  }
}

size_t register_expert_tensors(const Model& m) {
  std::vector<std::pair<uintptr_t, uintptr_t>> r;
  for (const Layer& l : m.layers)
    if (l.loaded && l.gate_exps)
      for (const ggml_tensor* t : {l.gate_exps, l.up_exps, l.down_exps})
        r.push_back({(uintptr_t)t->data & ~(uintptr_t)4095, ((uintptr_t)t->data + ggml_nbytes(t) + 4095) & ~(uintptr_t)4095});
  std::sort(r.begin(), r.end());
  std::vector<std::pair<uintptr_t, uintptr_t>> merged;
  for (auto& x : r)
    if (!merged.empty() && x.first <= merged.back().second) merged.back().second = std::max(merged.back().second, x.second);
    else merged.push_back(x);
  size_t n = 0;
  for (auto& [a, b] : merged) {
    if (hipHostRegister((void*)a, b - a, hipHostRegisterReadOnly | hipHostRegisterPortable) != hipSuccess) {
      (void)hipGetLastError();
      for (auto& [a2, b2] : merged) {  // undo the ones that worked
        if (a2 == a) break;
        (void)hipHostUnregister((void*)a2);
      }
      return 0;
    }
    n += b - a;
  }
  return n;
}

ExpertLoader::ExpertLoader(const Model& m_, std::vector<ExpertPool*> pools_, bool direct) : m(m_), pools(std::move(pools_)) {
  for (int L = 0; L < (int)m.layers.size(); L++)
    if (m.layers[L].loaded && m.layers[L].gate_exps) layers.push_back(L);
  queues.resize(pools.size());
  if (direct)
    reg_thread = std::thread([this] {
      const auto t0 = std::chrono::steady_clock::now();
      const size_t n = register_expert_tensors(m);
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (n) {
        fprintf(stderr, "# expert loader: %.1f GB of expert tensors registered in %.1f s (direct loads from now)\n", n / 1e9, s);
        registered = true;
      } else {
        fprintf(stderr, "# expert loader: registering the expert tensors failed; pinned loads\n");
      }
    });
}

bool ExpertLoader::wait_direct() {
  if (reg_thread.joinable()) reg_thread.join();
  return registered;
}

void ExpertLoader::start_test(int n, double rate) {
  test_n = n;
  test_rate = rate / pools.size();
  for (int d = 0; d < (int)pools.size(); d++) workers.emplace_back([this, d] { work(d); });
}

void ExpertLoader::submit(const Request& r) {
  if (workers.empty())
    for (int d = 0; d < (int)pools.size(); d++) workers.emplace_back([this, d] { work(d); });
  {
    std::lock_guard<std::mutex> lk(mu);
    queues[r.die].push_back(r);
  }
  cv.notify_all();
}

void ExpertLoader::work(int d) {
  ExpertPool& p = *pools[d];
  int lo, hi;
  CHECK(hipSetDevice(p.dev));
  CHECK(hipDeviceGetStreamPriorityRange(&lo, &hi));
  const size_t bytes = (size_t)p.view.gate.N * p.view.gate.K / 32 * 17;
  LoadStage st = load_stage_create(p.dev, bytes, lo);
  hipEvent_t fin;  // a served request's completion (blocking: the thread sleeps)
  CHECK(hipEventCreateWithFlags(&fin, hipEventDisableTiming | hipEventBlockingSync));
  std::mt19937 rng(1234 + d);
  const auto t0 = std::chrono::steady_clock::now();
  long n = 0;
  while (!stop) {
    Request r;
    if (test_rate > 0) {
      const auto due = t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(n / test_rate));
      if (std::chrono::steady_clock::now() < due) {
        std::this_thread::sleep_until(std::min(due, std::chrono::steady_clock::now() + std::chrono::milliseconds(20)));
        continue;
      }
      r = {d, p.n_slots - test_n + (int)(rng() % test_n), layers[rng() % layers.size()], (int)(rng() % m.hp.n_expert)};
    } else {
      std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&] { return stop || !queues[d].empty(); });
      if (stop) break;
      r = queues[d].front();
      queues[d].pop_front();
    }
    load_expert(st, p, r.slot, m.layers[r.layer], r.expert, registered);
    n++;
    n_loads++;
    if (test_rate == 0) {
      CHECK(hipEventRecord(fin, st.s));
      CHECK(hipEventSynchronize(fin));
      std::lock_guard<std::mutex> lk(mu);
      done.push_back(r);
    }
  }
  (void)hipEventDestroy(fin);
  load_stage_free(st);
}

std::vector<ExpertLoader::Request> ExpertLoader::take_done() {
  std::lock_guard<std::mutex> lk(mu);
  std::vector<Request> r;
  r.swap(done);
  return r;
}

ExpertLoader::~ExpertLoader() {
  stop = true;
  cv.notify_all();
  for (auto& t : workers) t.join();
  if (reg_thread.joinable()) reg_thread.join();
}
