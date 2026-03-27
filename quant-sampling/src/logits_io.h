#pragma once

#include <cstdint>
#include <string>
#include <vector>

static constexpr char     QMLOG_MAGIC[]   = "qmlogits";
static constexpr uint32_t QMLOG_VERSION   = 1;
static constexpr size_t   QMLOG_HEADER_SZ = 72;

struct qmlog_header {
    uint32_t version   = QMLOG_VERSION;
    int32_t  n_vocab   = 0;
    int32_t  n_tokens  = 0;
    int32_t  n_prompt  = 0;
    float    temp      = 0.0f;
    float    top_p     = 0.0f;
    int32_t  top_k     = 0;
    uint32_t seed      = 0;
};

struct qmlog_file {
    qmlog_header             header;
    std::vector<int32_t>     tokens;  // [n_tokens]
    std::vector<float>       logits;  // [(n_tokens - 1) * n_vocab]
};

// Write a .qmlog file. Returns true on success.
bool qmlog_write(const std::string & path, const qmlog_file & f);

// Read a .qmlog file. Returns true on success.
bool qmlog_read(const std::string & path, qmlog_file & f);

// Read only the header + tokens (skip logits). Returns true on success.
bool qmlog_read_tokens(const std::string & path, qmlog_file & f);
