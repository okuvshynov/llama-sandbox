// Background expert loads into expert pool slots: the adaptive cache's load path (PLAN.md
// "Adaptive expert cache", phases 2-3; README step 30).
//
// A load copies one expert's gate, up and down (3 x 6.27 MB of ggml mxfp4 blocks) from the
// model's mmap to a die and unpacks it into a slot (fp4_unpack), on a low-priority stream.
// Two paths:
//   direct  the expert tensors are registered with HIP (read-only), so the copy engine reads the
//           page cache itself: ~1 ms of CPU per load (API calls), no memory traffic on the CPU
//   pinned  a host memcpy into pinned staging first (pool_load's path): ~2.2 ms of CPU and
//           2 x 18.8 MB of memory traffic per load, which slows the CPU expert kernel ~7% at
//           100 loads/s against ~1% direct
// Registering the 289 GB of expert tensors takes ~40 s, so it runs on a background thread and
// the loaders use the pinned path until it is done.
//
// At most 2 matrices are in flight per loader: copies queued on a die delay its other copies
// (the per-layer staging) whatever the stream's priority.
#pragma once

#include "expert_pool.h"
#include "model.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

// One loader's staging: pinned host and device buffers (two each, used in turn) and a stream.
struct LoadStage {
  int dev = -1;
  hipStream_t s = nullptr;
  uint8_t* host[2] = {nullptr, nullptr};
  uint8_t* dstage[2] = {nullptr, nullptr};
  hipEvent_t free[2] = {nullptr, nullptr};  // the copy and unpack out of buffer b are done
  int next = 0;
};
// priority: hipDeviceGetStreamPriorityRange's least (numerically greatest) for background loads
LoadStage load_stage_create(int dev, size_t matrix_bytes, int priority);
void load_stage_free(LoadStage& st);
// Queue the load of expert e of layer l into `slot` of `pool` (current device = st.dev); returns
// once the host side is done (the GPU side completes in order on st.s). direct: l's expert
// tensors are registered (register_expert_tensors).
void load_expert(LoadStage& st, ExpertPool& pool, int slot, const Layer& l, int e, bool direct);

// Registers every loaded layer's expert tensors with HIP (read-only; neighbouring tensors'
// page-aligned ranges merged). Returns the bytes registered, 0 on failure.
size_t register_expert_tensors(const Model& m);

// One loader thread per die, each with its own LoadStage, fed by a queue.
class ExpertLoader {
 public:
  struct Request {
    int die;  // index into pools
    int slot, layer, expert;
  };
  // direct: register the expert tensors on a background thread and use the direct path once
  // they are (pinned until then); false: always pinned
  ExpertLoader(const Model& m, std::vector<ExpertPool*> pools, bool direct);
  ~ExpertLoader();
  void submit(const Request& r);
  // test mode (README step 30): every die loads random experts into the last n slots of its
  // pool at `rate` loads/s in all, until destroyed; nothing reads those slots
  void start_test(int n, double rate);
  // blocks until registering is over (true: direct loads from now on)
  bool wait_direct();
  long loads() const { return n_loads.load(); }
  // requests whose copy and unpack have finished since the last call (served from the queue)
  std::vector<Request> take_done();
  bool direct_ready() const { return registered.load(); }

 private:
  const Model& m;
  std::vector<ExpertPool*> pools;
  std::vector<int> layers;  // layers with experts
  std::vector<std::thread> workers;
  std::thread reg_thread;
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::deque<Request>> queues;
  std::vector<Request> done;  // finished, not yet taken (guarded by mu)
  std::atomic<bool> stop{false}, registered{false};
  std::atomic<long> n_loads{0};
  double test_rate = 0;  // loads/s per die; 0 = serve the queue
  int test_n = 0;
  void work(int die);
};
