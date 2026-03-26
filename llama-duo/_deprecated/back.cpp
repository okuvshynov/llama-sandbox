#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include <common.h>
#include <llama.h>
#include <nlohmann/json.hpp>
#include <httplib.h>

#include "utils.h"

namespace llama_duo
{

struct config
{
    std::string host;
    int32_t     port;

    std::string model_path;
    uint32_t    n_batch;
    uint32_t    n_ctx;
    uint32_t    n_threads;
    uint32_t    n_gpu_layers;
    size_t      n_ahead;      // wait after n_ahead non-validated tokens.
};

config gen_config(int argc, char ** argv)
{
    config res = 
    {
        /* host         = */ "localhost",
        /* port         = */ 5555,

        /* model_path   = */ "",
        /* n_batch      = */ 512,
        /* n_ctx        = */ 4096,
        /* n_threads    = */ 16,
        /* n_gpu_layers = */ 0,

        /* n_ahead      = */ 16
    };
    parser<config> p;
    // main server endpoint to connect to
    p.add_option({"--host", "-h"},                             &config::host);
    p.add_option({"--port", "-p"},                             &config::port);

    // llama options
    p.add_option({"--model", "-m"},                            &config::model_path);
    p.add_option({"--batch_size", "--batch-size", "-b"},       &config::n_batch);
    p.add_option({"--n_ctx", "--n-ctx", "-c"},                 &config::n_ctx);
    p.add_option({"--threads", "-t"},                          &config::n_threads);
    p.add_option({"--n_gpu_layers", "--n-gpu-layers", "-ngl"}, &config::n_gpu_layers);
    p.add_option({"--n_ahead", "--n-ahead", "-na"},            &config::n_ahead);

    if (0 != p.parse_options(argc, argv, res))
    {
        exit(-1);
    }

    return res;
}

using json = nlohmann::json;

using llama_tokens = std::vector<llama_token>;

int loop(config conf)
{
    using namespace std::chrono_literals;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = conf.n_gpu_layers;

    llama_model * model = llama_load_model_from_file(conf.model_path.c_str(), model_params);
    if (model == nullptr)
    {
        return 1;
    }

    httplib::Client http_client(conf.host, conf.port);
    http_client.set_keep_alive(true);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_batch   = conf.n_batch;
    ctx_params.n_ctx     = conf.n_ctx;
    ctx_params.n_threads = conf.n_threads;

    llama_context * llama_ctx = llama_new_context_with_model(model, ctx_params);
    llama_batch         batch = llama_batch_init(conf.n_batch, 0, 1);
    
    llama_tokens curr, updated; // empty 
    size_t n_not_rejected   = 0;
    size_t n_approved       = 0;
    size_t n_prefix         = 0;
    uint32_t crc32_approved = 0;  

    while (true)
    {
        try
        {
            json req;
            
            // curr[:n_approved] was confirmed by main model. However, we need to make sure we 
            // are working on the same sequence. So we pass the length of the prefix (=n_prefix) and 
            // its crc32 checksum. Main server will check that it matches ground truth sequence.
            // Alternative way to handle this would be to have some sort of query_id or session_id.
            // 
            // At small context lengths delta passing wouldn't be needed and 
            // we could just pass entire speculation. For longer conversation
            // with large contexts would become slow to pass
            // entire token lists back and forth.

            req["candidate"]     = llama_tokens(curr.begin() + n_approved, curr.end());
            // what's the offset of the tokens we pass.
            req["n_prefix"]     = n_approved;
            // what's the checksum of the omitted prefix.
            req["crc32_prefix"] = crc32_approved;

            // TODO: should this be some long poll when we wait?
            auto res = http_client.Post("/hint", req.dump(), "application/json");
            if (res)
            {
                json res_j = json::parse(res->body);
                
                // new candidate
                updated  = res_j["candidate"].get<llama_tokens>();

                // at what offset does it start?
                n_prefix = res_j["n_prefix"].get<size_t>();

                // remove everything non-matching. 
                // TODO: this will probably remove everything or nothing
                curr.erase(curr.begin() + n_prefix, curr.end());
                curr.insert(curr.end(), updated.begin(), updated.end());

                // how many tokens 'matched'. Not all of them were approved,
                // but none were rejected by main model yet.
                // n_not_rejected is relative to n_prefix, so total number of non-rejected tokens is 
                // n_not_rejected + n_prefix.
                n_not_rejected  = res_j["n_not_rejected"].get<size_t>();

                // How many tokens were validated by main model
                n_approved = res_j["n_approved"].get<size_t>();

                // what's the checksum of that validated prefix
                crc32_approved = res_j["crc32_approved"].get<uint32_t>();
            }
        }
        catch(const std::exception& e)
        {
            log::error("%s", e.what());
            std::this_thread::sleep_for(500ms);
            continue;
        }

        if (curr.size() == 0 || updated.size() == 0 || (n_approved > 0 && curr.size() > n_approved + conf.n_ahead))
        {
            log::info(
                "waiting; curr.size() = %zu, updated.size() = %zu, n_approved = %zu",
                curr.size(),
                updated.size(),
                n_approved
            );
            std::this_thread::sleep_for(500ms);
            continue;
        }

        // remove the mismatched entries from KV cache
        llama_kv_cache_seq_rm(llama_ctx, 0, n_prefix + n_not_rejected, -1);

        // generate at least one
        if (n_prefix + n_not_rejected == curr.size())
        {
            n_not_rejected -= 1;
        }

        // batched evaluation. Only last token produces logits.
        auto bsz = conf.n_batch;
        for (size_t i = n_prefix + n_not_rejected; i < curr.size();)
        {
            llama_batch_clear(batch);
            size_t j;
            for (j = 0; j < bsz && i + j < curr.size(); j++)
            {
                llama_batch_add(batch, curr[i + j], i + j, { 0 }, false);
            }
            if (i + j == curr.size())
            {
                batch.logits[batch.n_tokens - 1] = true;
            }
            if (llama_decode(llama_ctx, batch) != 0)
            {
                log::error("%s: llama_decode() failed\n", __func__);
                continue;
            }
            i += j;
        }

        // pick greedily
        auto next_tokens = greedy_tokens(model, llama_ctx, batch.n_tokens - 1, batch.n_tokens);
        if (next_tokens.size() != 1)
        {
            log::error("invalid next tokens size");
            continue;
        }

        curr.push_back(next_tokens[0]);
    }

    llama_batch_free(batch);
    llama_free(llama_ctx);
    llama_free_model(model);
    return 0;
}

} // namespace llama_duo

int main(int argc, char ** argv)
{
    int res = 0;
    llama_backend_init();
    auto conf = llama_duo::gen_config(argc, argv);

    res = loop(conf);

    llama_backend_free();
    return res;
}
