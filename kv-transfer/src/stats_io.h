#pragma once

#include <cstdint>
#include <string>
#include <vector>

static constexpr char     STATS_MAGIC[]   = "qmstats\0";
static constexpr uint32_t STATS_VERSION   = 1;

// Per-token stats file — compact output from target/handoff runs.
//
// Binary layout (32-byte header + data):
//   [0..7]   magic "qmstats\0"
//   [8..11]  version (uint32 = 1)
//   [12..15] n_gen (int32)
//   [16..19] n_prompt (int32)
//   [20..23] temp (float)
//   [24..31] reserved (8 bytes, zero)
//   [32..]   kl[n_gen] (float32), then top1_match[n_gen] (uint8)

struct stats_file {
    uint32_t version  = STATS_VERSION;
    int32_t  n_gen    = 0;
    int32_t  n_prompt = 0;
    float    temp     = 0.0f;
    std::vector<float>   kl;          // [n_gen] per-token KL divergence
    std::vector<uint8_t> top1_match;  // [n_gen] 1 if argmax agrees, 0 otherwise
};

bool stats_write(const std::string & path, const stats_file & s);
bool stats_read(const std::string & path, stats_file & s);
