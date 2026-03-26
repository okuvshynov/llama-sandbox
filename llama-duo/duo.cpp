#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <common.h>
#include <llama.h>

//#include "utils.h"

namespace llama_duo
{

void dbg_color(const std::string & s, const std::string & fg)
{
    static const std::string kReset = "\033[0m";
    static const std::string bold[] = { "", "\033[1m" };
    static size_t index = 0;
    std::cout << bold[index] << fg << s << kReset << std::flush;
    index = 1 - index;
}

void dbg_accepted(const std::string & accepted)
{
    static const std::string kGreen = "\033[32m";
    dbg_color(accepted, kGreen);
}

void dbg_not_matched(const std::string & accepted)
{
    dbg_color(accepted, "");
}

void dbg_rejected(const std::string & rejected)
{
    static const std::string kRed = "\033[31m";
    dbg_color(rejected, kRed);
}

template<typename iter_t>
std::string to_string(llama_context * ctx, iter_t from, iter_t to)
{
    std::string res = "";
    for (auto it = from; it != to; ++it)
    {
        res += llama_token_to_piece(ctx, *it);
    }
    return res;
}

std::vector<llama_token> greedy_tokens(
        llama_model * model,
        llama_context * ctx,
        int32_t from_idx,
        int32_t to_idx)
{
    auto n_vocab = llama_n_vocab(model);
    std::vector<llama_token> res;
    if (n_vocab <= 0)
    {
        return res;
    }

    for (int idx = from_idx; idx < to_idx; idx++)
    {
        auto * logits  = llama_get_logits_ith(ctx, idx);
        llama_token new_token_id = 0;
        for (llama_token token_id = 1; token_id < n_vocab; token_id++)
        {
            if (logits[token_id] > logits[new_token_id])
            {
                new_token_id = token_id;
            }
        }

        res.push_back(new_token_id);
    }
    return res;
}

using llama_tokens = std::vector<llama_token>;

enum Turn
{
    NONE = 0,
    SPEC = 1,
    MAIN = 2
};

struct shared_context
{
    llama_tokens candidate;
    std::mutex   mtx;
    bool         done = false;
    Turn         turn = NONE;
    std::condition_variable cv;
};

template<typename iter_t>
static int decode(llama_context * ctx, iter_t from, iter_t to, int offset, bool all_logits, llama_batch & batch)
{
    llama_batch_clear(batch);
    size_t i = offset;
    for (auto it = from; it != to; ++it)
    {
        llama_batch_add(batch, *it, i++, { 0 }, all_logits);
    }
    batch.logits[batch.n_tokens - 1] = true;
    int res = 0;
    if (llama_decode(ctx, batch) != 0)
    {
        fprintf(stderr, "llama_decode() failed: n_tokens=%d\n", batch.n_tokens - 1);
        res = 1;
    }
    return res;
}

static void speculation(
    llama_model    * model,
    llama_context  * ctx,
    shared_context * sctx,
    const llama_tokens & input,
    size_t n_draft) 
{
    llama_batch batch = llama_batch_init(512, 0, 1);
    decode(ctx, input.begin(), input.end(), 0, false, batch);

    int logit_idx = input.size() - 1;
    llama_tokens local = input, shared;
    size_t match_len;

    while (true) 
    {
        {
            std::unique_lock<std::mutex> lock(sctx->mtx);
            sctx->cv.wait(lock, [&sctx] { return sctx->turn == Turn::SPEC || sctx->done; });
            if (sctx->done)
            {
                break;
            }
            shared = sctx->candidate;
            sctx->turn = Turn::NONE;
        }

        bool match = true;
        match_len = local.size() - 1;
        for (size_t i = 0; i < std::min(shared.size(), local.size()); i++)
        {
            if (shared[i] != local[i])
            {
                match = false;
                match_len = i;
                llama_kv_cache_seq_rm(ctx, 0, i, -1);
                break;
            }
        }
        if (!(match && shared.size() < local.size())) 
        {
            local = shared;
        }

        for (size_t i = 0; i < n_draft; i++)
        {
            decode(ctx, local.begin() + match_len, local.end(), match_len, false, batch);
            logit_idx = local.size() - match_len - 1;
            auto next_tokens = greedy_tokens(model, ctx, logit_idx, logit_idx + 1);
            match_len = local.size();
            local.push_back(next_tokens[0]);
        }

        {
            std::unique_lock<std::mutex> lock(sctx->mtx);
            sctx->candidate = local;
            sctx->turn = Turn::MAIN;
            sctx->cv.notify_one();
        }
    }

    llama_batch_free(batch);
}

static void target(
    llama_model    * model,
    llama_context  * ctx,
    shared_context * sctx,
    const llama_tokens & input,
    size_t n_predict)
{
    dbg_not_matched(to_string(ctx, input.begin(), input.end()));

    llama_batch batch = llama_batch_init(512, 0, 1);
    decode(ctx, input.begin(), input.end(), 0, false, batch);

    size_t n_accepted = input.size();

    int logits_from = input.size() - 1;
    int logits_to   = input.size();

    llama_tokens input_seq, next_tokens;
    input_seq.push_back(input.back());

    auto start_us = ggml_time_us();

    while (n_accepted < n_predict + input.size())
    {
        next_tokens = greedy_tokens(model, ctx, logits_from, logits_to);

        size_t next_tokens_pos = n_accepted;
        // we always accept at least one new token
        n_accepted += 1;
        size_t n_match = 0;
        while (n_match + 1 < input_seq.size() && next_tokens[n_match] == input_seq[n_match + 1])
        {
            n_match++;
        }
        n_accepted += n_match;
        next_tokens.erase(next_tokens.begin() + n_match + 1, next_tokens.end());
        llama_kv_cache_seq_rm(ctx, 0, n_accepted - 1, -1);

        bool eog = false;
        for (size_t i = 0; i < next_tokens.size(); i++)
        {
            // TODO: what should we do here, is this correct
            if (next_tokens[i] == llama_token_eos(model) || llama_token_is_eog(model, next_tokens[i]))
            {
                eog = true;
                next_tokens.erase(next_tokens.begin() + i, next_tokens.end());
                break;
            }
        }

        {
            std::unique_lock<std::mutex> lock(sctx->mtx);
            sctx->cv.wait(lock, [&sctx] { return sctx->turn == Turn::MAIN; });
            auto & spec = sctx->candidate;
            size_t n_match = 0;
            while (n_match < next_tokens.size()
                && n_match + next_tokens_pos < spec.size()
                && next_tokens[n_match] == spec[n_match + next_tokens_pos])
            {
                n_match++;
            }

            dbg_accepted(to_string(ctx, spec.begin() + next_tokens_pos, spec.begin() + next_tokens_pos + n_match));
            if (n_match != next_tokens.size())
            {
                //dbg_rejected(to_string(ctx, spec.begin() + next_tokens_pos + n_match, spec.end()));
                dbg_not_matched(to_string(ctx, next_tokens.begin() + n_match, next_tokens.end()));
                spec.erase(spec.begin() + next_tokens_pos, spec.end());
                for (const auto tok: next_tokens)
                {
                    spec.push_back(tok);
                }
            }
            input_seq.assign(spec.begin() + n_accepted - 1, spec.end());
            sctx->turn = Turn::SPEC;
            sctx->cv.notify_one();
        }

        if (n_accepted >= n_predict + input.size() || eog)
        {
            break;
        }

        decode(ctx, input_seq.begin(), input_seq.end(), n_accepted - 1, true, batch);

        logits_from = 0;
        logits_to   = input_seq.size();
    }

    double dur_s  = 1.0e-6 * (ggml_time_us() - start_us);
    size_t tokens = n_accepted - input.size(); 
    
    dbg_not_matched("\n");
    std::cerr << "tokens: " << tokens << " tps: " << tokens / dur_s << std::endl;
    {
        std::lock_guard<std::mutex> _lock(sctx->mtx);
        sctx->done = true;
    }

    llama_batch_free(batch);
}

} // llama_duo

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false)
    {
        return 1;
    }

    if (params.seed == LLAMA_DEFAULT_SEED)
    {
        params.seed = time(NULL);
    }
    
    // TODO: hacky: we use rpc_servers for draft model only
    std::string draft_rpc = params.rpc_servers;
    params.rpc_servers = "";

    llama_backend_init();
    llama_numa_init(params.numa);

    // main model and context
    llama_init_result main_init = llama_init_from_gpt_params(params);
    llama_model * model = main_init.model;
    llama_context * ctx = main_init.context;

    llama_duo::llama_tokens input = llama_tokenize(ctx, params.prompt, true);

    params.model = params.model_draft;
    params.n_gpu_layers = params.n_gpu_layers_draft;
    if (params.n_threads_draft > 0) 
    {
        params.n_threads = params.n_threads_draft;
    }
    params.n_threads_batch = params.n_threads_batch_draft;

    params.rpc_servers = draft_rpc;
    llama_init_result draft_init = llama_init_from_gpt_params(params);
    // draft model and contexts.
    llama_model * draft_model = draft_init.model;
    llama_context * draft_ctx = draft_init.context;
    
    llama_duo::shared_context sctx;
    sctx.candidate = input;
    sctx.turn = llama_duo::Turn::SPEC;

    std::thread spec_thread = std::thread(llama_duo::speculation, draft_model, draft_ctx, &sctx, input, params.n_draft);
    target(model, ctx, &sctx, input, params.n_predict);
    spec_thread.join();
    
    llama_free(ctx);
    llama_free(draft_ctx);
    llama_free_model(model);
    llama_free_model(draft_model);
    llama_backend_free();

    return 0;
}
