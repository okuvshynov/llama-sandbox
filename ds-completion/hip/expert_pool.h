// Routed experts resident on GPUs one expert at a time (PLAN.md "Expert placement and cache"):
// each die holds a pool of expert-sized slots, and a placement says which (layer, expert) pairs
// live in which die's slots; every other pair runs on the CPU.
//
// A slot holds one expert's gate, up and down matrices in fp4_moe's layout (codes + scales), so
// fp4_moe runs on a pool directly with slot numbers as its ids (pool.view).
//
// The placement file (scripts/route_stats.py --emit-placement) is a ranked list of pairs, most
// used first; place_experts() keeps the top ones that fit, spreading each layer's experts over
// the dies in turn so that one layer's hits split evenly.
#pragma once

#include "fp4_moe.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct ExpertPool {
  int dev = -1;
  int n_slots = 0;
  size_t slot_bytes = 0;
  uint8_t* base = nullptr;     // n_slots * slot_bytes on dev
  uint8_t* stage = nullptr;    // raw mxfp4 blocks of one matrix, on dev
  // pinned host staging, two buffers in turn: copying straight from the model's mmap (pageable,
  // file-backed) intermittently crashed the GPU's copy engine ("page not present") when 4 dies
  // loaded at once; a host memcpy into pinned memory first avoids it
  uint8_t* host_stage[2] = {nullptr, nullptr};
  hipEvent_t host_free[2] = {nullptr, nullptr};  // the copy out of that buffer has finished
  int next_buf = 0;
  GpuExperts view;             // the three matrices, addressed by slot
};

// bytes of one slot for experts of n_embd E and n_ff_exp F (gate, up, down; 256-aligned parts)
size_t pool_slot_bytes(int E, int F);
// allocates n_slots on dev (current device is restored)
ExpertPool pool_create(int dev, int n_slots, int E, int F);
// slots that fit in dev's free memory, keeping `reserve` bytes free
int pool_slots_that_fit(int dev, int E, int F, size_t reserve);
// copy one expert into a slot: gate/up/down are host pointers to that expert's ggml mxfp4 blocks
// (a layer tensor's data + e * nb[2]); s must be a stream on pool.dev (current device = pool.dev)
void pool_load(ExpertPool& pool, int slot, const void* gate, const void* up, const void* down, hipStream_t s);
void pool_free(ExpertPool& pool);

// The ranked (layer, expert) list of a placement file.
std::vector<std::pair<int, int>> read_placement(const std::string& path);

// Where every (layer, expert) pair lives: die >= 0 and its slot, or die -1 (CPU).
struct Placement {
  int n_expert = 0;
  std::vector<int16_t> die, slot;  // [layer * n_expert + expert]
  std::vector<int> used;           // slots used per die
  int at(int L, int e) const { return die[(size_t)L * n_expert + e]; }
};

// Fill the dies' slots (capacity[d] each) with the ranked pairs, skipping layers not in
// `layers_loaded`; each layer's pairs go to the dies in turn, skipping full ones.
Placement place_experts(const std::vector<std::pair<int, int>>& ranked, int n_layer, int n_expert,
                        const std::vector<int>& capacity, const std::vector<bool>& layers_loaded);
