#pragma once

#include <cstdint>
#include <string>
#include <vector>

// "lkldtopk" v1 file: token sequence + per-position top-K logits with a full-vocab
// log-sum-exp normalizer. The reference reader lives in inspect.py — keep both in sync.

static constexpr char     LKLD_MAGIC[]  = "lkldtopk";
static constexpr uint32_t LKLD_VERSION  = 1;

// Per-position logit record. LSE over the full vocab = max_logit + lse_rest;
// logprob of stored entry k = logits[k] - max_logit - lse_rest.
struct lkld_position {
    float                max_logit = 0.0f;  // max over full vocab (== logits[0])
    float                lse_rest  = 0.0f;  // log sum exp(logit - max_logit) over full vocab
    std::vector<int32_t> ids;               // [top_k], sorted by logit desc, ties id asc
    std::vector<float>   logits;            // [top_k], raw fp32 logits
};

struct lkld_seq {
    std::string                label;       // prompt source ("inline" or file path)
    int32_t                    n_prompt = 0;
    int32_t                    n_total  = 0; // prompt + generated
    std::vector<int32_t>       tokens;      // [n_total]
    std::vector<lkld_position> positions;   // [n_scored], n_scored == n_total in v1
};

struct lkld_file {
    int32_t                n_vocab = 0;
    int32_t                top_k   = 0;
    std::string            model_desc;      // model path + llama_model_desc()
    std::vector<lkld_seq>  seqs;
};

// Write a v1 .bin file. Returns true on success.
bool lkld_write(const std::string & path, const lkld_file & f);
