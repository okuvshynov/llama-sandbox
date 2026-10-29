// Adaptive expert cache (PLAN.md "Adaptive expert cache", phase 3b; README step 32): the GPU
// expert pools follow the experts in use instead of a fixed placement.
//
// Policy (phase 1's simulation, README step 29): count each (layer, expert)'s use over the last
// `window` forwarded tokens (rejected drafts included); every `period` tokens the most-used
// experts not on a GPU replace the least-used ones on a GPU, a swap only if the newcomer was
// used more (ties: least recently used first). Loads start at up to `rate` per second (a token
// bucket); 0 = the static placement.
//
// Table protocol (the runtime calls step() between forwards, when no GPU work reads a pool):
//   1. loads that finished are published: their (layer, expert) -> (die, slot) entry is set;
//   2. a swap first removes the victim from the table (it runs on the CPU from the next forward),
//      the table is uploaded, and only then the loader may overwrite the victim's slot;
//   3. the newcomer is published only after its copy and unpack are done (step 1 of a later
//      boundary).
// So no kernel ever reads a slot being written, and an entry never points at a half-loaded
// expert. With --canon the output does not depend on any of this; without it, it depends on
// when loads finish.
#pragma once

#include "expert_loader.h"
#include "expert_pool.h"
#include "kernels.h"  // MoeList

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

class ExpertCache {
 public:
  struct Options {
    double rate = 0;    // loads per second
    int period = 16;    // tokens between rebalances
    int window = 256;   // tokens of usage counted
    // loads go straight from the registered model mapping (expert_loader.h); registering takes
    // ~40 s, and until then no load starts (pinned loads would cost ~9%, README step 30).
    // wait = block in the constructor until it is done (for measurements)
    bool wait = false;
  };
  // table: the host copy of the runtime's [n_layer][n_expert] table (die << 16 | slot, or -1),
  // filled by the static placement; d_table its device copy (uploaded by step())
  ExpertCache(const Model& m, std::vector<ExpertPool*> pools, std::vector<int>& table, int* d_table, int n_expert, Options o);
  ~ExpertCache();

  // the dispatch lists of one layer of the current forward (host copies, M tokens): counts use
  void observe(int L, const MoeList* lists, int M, int n_dies);
  // a step boundary: the current forward's M tokens enter the window; publish finished loads,
  // rebalance when due, start loads within the rate
  void step(int M);

  struct Stats { long loads_started = 0, loads_published = 0, rebalances = 0, swaps_planned = 0; };
  Stats stats() const { return st; }

 private:
  const int n_expert, n_pairs, n_dies;
  Options opt;
  std::vector<int>& table;
  int* d_table;
  std::vector<ExpertPool*> pools;
  std::vector<std::vector<int>> slot_owner;  // [die][slot]: pair key, or -1 while loading
  std::vector<int8_t> loading;               // [pair]: a load into a slot is in flight
  std::vector<int> count;                    // [pair]: uses in the window
  std::vector<int64_t> last_use;             // [pair]: token index of the last use
  std::deque<std::pair<int, std::vector<int>>> hist;  // (tokens, keys) per forward, in the window
  std::vector<int> cur;                      // keys of the current forward
  int hist_tokens = 0;
  int64_t tokens = 0, next_rebalance = 0;
  struct Swap { int pair, victim; };
  std::deque<Swap> plan;                     // planned by the last rebalance, not yet started
  double bucket = 0;
  std::chrono::steady_clock::time_point last_tick;
  bool table_dirty = false;
  std::unique_ptr<ExpertLoader> loader;
  Stats st;
  void rebalance();
};
