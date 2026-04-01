#pragma once

#include <cstdint>
#include <string>
#include <vector>

static constexpr char     TRACE_MAGIC[]   = "qmlogits";
static constexpr uint32_t TRACE_VERSION   = 3;

// Per-prompt data section.
struct trace_entry {
    std::string          path;              // source file path (e.g. "math/01.txt")
    int32_t              n_tokens = 0;      // total tokens (prompt + generated)
    int32_t              n_prompt = 0;      // prompt-only token count
    std::vector<int32_t> tokens;            // [n_tokens]
    std::vector<float>   logits;            // [(n_tokens - 1) * n_vocab]
};

// Multi-prompt file.
struct trace_file {
    // global header fields
    uint32_t version    = TRACE_VERSION;
    int32_t  n_vocab    = 0;
    int32_t  n_prompts  = 0;
    float    temp       = 0.0f;
    float    top_p      = 0.0f;
    int32_t  top_k      = 0;
    uint32_t seed       = 0;
    // per-prompt data
    std::vector<trace_entry> prompts;
};

// Write a multi-prompt .bin file (v3). Returns true on success.
bool trace_write(const std::string & path, const trace_file & f);

// Read a v3 .bin file. Returns true on success.
bool trace_read(const std::string & path, trace_file & f);

// Read only the header + tokens for all prompts (skip logits). Returns true on success.
bool trace_read_tokens(const std::string & path, trace_file & f);
