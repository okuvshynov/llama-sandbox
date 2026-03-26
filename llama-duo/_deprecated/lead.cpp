#include <cstdint>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include <common.h>
#include <llama.h>
#include <nlohmann/json.hpp>
#include <httplib.h>

#include "utils.h"

namespace llama_duo
{

using llama_tokens = std::vector<llama_token>;

struct spec_context
{
    llama_tokens candidate;          // current shared candidate
    size_t       n_approved     = 0; // how many were validated by main model
    uint32_t     crc32_approved = 0; // crc32 checksum of approved part 
    std::mutex   mtx;
};

struct query_context
{
    std::string  prompt;       // original input
    llama_tokens last_session; // prompt + output for the latest completion. 
    llama_tokens output;       // generated output w/o prompt
    llama_batch  batch;        // llama batch we reuse
    size_t       n_len;        // how many tokens at most (prompt + generated)
    std::mutex   mtx;          // ensure we process 1 query at a time
};

struct config
{
    std::string host;          // to listen on. "0.0.0.0" is default. 
    int32_t     port;          // 5555 is default.

    std::string model_path;    // path to gguf file
    uint32_t    n_batch;       // batch size
    uint32_t    n_ctx;         // context size (n_len must be <= n_ctx)
    uint32_t    n_threads;     // how many threads to use for CPU eval.
    uint32_t    n_gpu_layers;  // how many layers to offload to GPU.

    std::string print_mode;    // how to print the output to stdout. 
                               // none      -- no text output
                               // all       -- everything including rejected tokens
                               // accepted  -- non-rejected tokens only
};

config gen_config(int argc, char ** argv)
{
    config res = 
    {
        /* host         = */ "0.0.0.0",
        /* port         = */ 5555,

        /* model_path   = */ "",
        /* n_batch      = */ 512,
        /* n_ctx        = */ 4096,
        /* n_threads    = */ 16,
        /* n_gpu_layers = */ 0,
        /* print_mode   = */ "accepted"
    };
    parser<config> p;
    // server options
    p.add_option({"--host", "-h"},                             &config::host);
    p.add_option({"--port", "-p"},                             &config::port);

    // llama options
    p.add_option({"--model", "-m"},                            &config::model_path);
    p.add_option({"--batch_size", "--batch-size", "-b"},       &config::n_batch);
    p.add_option({"--n_ctx", "--n-ctx", "-c"},                 &config::n_ctx);
    p.add_option({"--threads", "-t"},                          &config::n_threads);
    p.add_option({"--n_gpu_layers", "--n-gpu-layers", "-ngl"}, &config::n_gpu_layers);
    p.add_option({"--print_mode", "--print-mode", "-pm"},      &config::print_mode);

    if (0 != p.parse_options(argc, argv, res))
    {
        exit(-1);
    }

    return res;
}

using json = nlohmann::json;

std::string llama3_instruct_fmt_msg(const json & j)
{
    std::ostringstream oss;
    oss << "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n";
    oss << j.value("system", "") << "<|eot_id|>\n";

    for (const auto& msg: j["messages"])
    {
        oss 
            << "<|start_header_id|>"
            << msg["role"].get<std::string>()
            << "<|end_header_id|>\n\n"
            << msg["content"].get<std::string>() << "<|eot_id|>";
    }

    oss << "<|start_header_id|>assistant<|end_header_id|>";
    return oss.str();
}

class llama_lead
{
  public:
    static std::unique_ptr<llama_lead> create(config conf);
    ~llama_lead();
    void serve();

  private:
    explicit llama_lead(config conf);
    int generate(const llama_tokens & tokens_list, size_t n_reuse = 0);

    const config    conf_;
    query_context   query_ctx_;
    spec_context    spec_ctx_;
    llama_model   * model_     = nullptr;
    llama_context * llama_ctx_ = nullptr;

    httplib::Server http_server_;
};

std::unique_ptr<llama_lead> llama_lead::create(config conf)
{
    auto self = std::unique_ptr<llama_lead>(new llama_lead(conf));

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = conf.n_gpu_layers;

    self->model_ = llama_load_model_from_file(conf.model_path.c_str(), model_params);

    if (self->model_ == nullptr)
    {
        log::fatal("Unable to load model from %s", conf.model_path.c_str());
        return nullptr;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_batch   = conf.n_batch;
    ctx_params.n_ctx     = conf.n_ctx;
    ctx_params.n_threads = conf.n_threads;
    self->llama_ctx_     = llama_new_context_with_model(self->model_, ctx_params);

    if (self->llama_ctx_ == nullptr)
    {
        log::fatal("Unable to create llama context");
        return nullptr;
    }

    return self;
}

llama_lead::llama_lead(config conf): conf_(conf)
{
}

llama_lead::~llama_lead()
{
    if (llama_ctx_ != nullptr)
    {
        llama_free(llama_ctx_);
    }
    if (model_ != nullptr)
    {
        llama_free_model(model_);
    }
}

void llama_lead::serve()
{
    http_server_.Post("/hint", [this](const httplib::Request & req, httplib::Response & res)
    {
        try
        {
            json res_j;
            auto req_j = json::parse(req.body);

            llama_tokens remote_candidate = req_j["candidate"];

            // offset based on what we approved in the past
            size_t       n_prefix         = req_j["n_prefix"];
            // crc32 checksum of non-passed prefix
            uint32_t     crc32_prefix     = req_j["crc32_prefix"]; 
            {
                std::lock_guard<std::mutex> _lock(spec_ctx_.mtx);
                auto& candidate = spec_ctx_.candidate;
                bool prefix_mismatch = false;

                // first, check prefix crc32. if we cannot match, need to return
                // entire candidate. likely it's a new query processing, etc.
                if (n_prefix > candidate.size())
                {
                    prefix_mismatch = true;
                }
                else
                {
                    uint32_t local_crc32_prefix = crc32(candidate.begin(), candidate.begin() + n_prefix);
                    if (local_crc32_prefix != crc32_prefix)
                    {
                        prefix_mismatch = true;
                    }
                }
                if (prefix_mismatch)
                {
                    res_j["candidate"]      = candidate;
                    // we pass entire candidate from position 0
                    res_j["n_prefix"]       = 0;
                    // how many do we not need to recompute on speculator
                    res_j["n_not_rejected"] = 0;
                    // how many were approved by main model
                    res_j["n_approved"]     = spec_ctx_.n_approved;
                    // crc checksum of the approved part
                    res_j["crc32_approved"] = spec_ctx_.crc32_approved;
                }
                else
                {
                    bool match = true;
                    size_t n_not_rejected = remote_candidate.size();
                    for (size_t i = 0; i < remote_candidate.size() && i + n_prefix < candidate.size(); i++)
                    {
                        if (candidate[i + n_prefix] != remote_candidate[i])
                        {
                            match = false;
                            n_not_rejected = i;
                            break;
                        }
                    }
                    if (match && candidate.size() < n_prefix + remote_candidate.size())
                    {
                        // append newly speculated tokens to local candidate
                        for (size_t i = 0; i < remote_candidate.size(); i++)
                        {
                            if (i + n_prefix < candidate.size())
                            {
                                continue;
                            }
                            candidate.push_back(remote_candidate[i]);
                        }
                    }
                    else
                    {
                        // we start at the same offset and copy local part of it
                        remote_candidate = llama_tokens(candidate.begin() + n_prefix, candidate.end());
                    }
                    // passing back the same offset
                    res_j["n_prefix"]       = n_prefix;
                    res_j["candidate"]      = remote_candidate;
                    res_j["n_not_rejected"] = n_not_rejected;
                    res_j["n_approved"]     = spec_ctx_.n_approved;
                    res_j["crc32_approved"] = spec_ctx_.crc32_approved;
                }
            }
            res.set_content(res_j.dump(), "application/json");
        }
        catch(const std::exception & e)
        {
            log::error("%s", e.what());
        }
    });

    http_server_.Post("/messages", [this](const httplib::Request & req, httplib::Response & res)
    {
        // we process one message at a time anyway for now.
        std::lock_guard<std::mutex> _lock(query_ctx_.mtx);

        try
        {
            auto req_j = json::parse(req.body);

            query_ctx_.prompt    = llama3_instruct_fmt_msg(req_j);
            size_t n_predict     = static_cast<size_t>(req_j.value("max_tokens", 1024));
            
            const auto t_start   = ggml_time_us();
            if (conf_.print_mode != "none")
            {
                dbg_not_matched(query_ctx_.prompt);
            }

            auto prompt = llama_tokenize(llama_ctx_, query_ctx_.prompt, false);
            
            // TODO: adding these two separate newlines for llama3 makes it match the last sequence
            // What's the right way to handle this?
            // prompt.push_back(198);
            // prompt.push_back(198);
            if (conf_.n_ctx < prompt.size())
            {
                log::error("context size %zu < prompt size %zu, unable to process prompt", conf_.n_ctx, prompt.size());
                // TODO: return error to client
                return;
            }
            if (conf_.n_ctx < n_predict + prompt.size())
            {
                log::warn("context not large enough, might trim output.");
                n_predict = conf_.n_ctx - prompt.size();
            }

            query_ctx_.batch = llama_batch_init(conf_.n_batch, 0, 1);
            query_ctx_.n_len = n_predict + prompt.size();
            // TODO come up with naming which would make it clear if something 
            // is a string or list of tokens
            query_ctx_.output.clear();

            // Init speculation context
            {
                std::lock_guard<std::mutex> _lock(spec_ctx_.mtx);
                spec_ctx_.candidate      = prompt;
                spec_ctx_.n_approved     = 0;
                spec_ctx_.crc32_approved = 0;
            }

            // check the match of the prefix for prompt + previously generated part
            // leave at least 1 input token in prompt. 
            // TODO: seems like there's some mismatch between \n\n in the middle and \n\n 
            // in end of string, so last reply of assistant has to be reprocessed. 
            size_t i = 0;
            for (i = 0; i < query_ctx_.last_session.size() && i + 1 < prompt.size(); i++)
            {
                if (query_ctx_.last_session[i] != prompt[i])
                {
                    break;
                }
            }
            // reusing cache for tokens [0; i)
            llama_kv_cache_seq_rm(llama_ctx_, 0, i, -1);

            if (generate(prompt, i) != 0)
            {
                log::error("generation failed");
            }

            const auto t_end = ggml_time_us();
            log::info("total generation time: %.3lf s", (t_end - t_start) / 1000000.0);;

            std::string output;
            for (auto tok: query_ctx_.output)
            {
                output += llama_token_to_piece(llama_ctx_, tok);
            }
            
            json res_j = { {"content",  {{"text", output}}} };
            res.set_content(res_j.dump(), "application/json");

            llama_batch_free(query_ctx_.batch);

            // together with query_ctx_.output this is 'previous session'
            query_ctx_.last_session = prompt;
            query_ctx_.last_session.insert(query_ctx_.last_session.end(), query_ctx_.output.begin(), query_ctx_.output.end());
        }
        catch (const std::exception & e)
        {
            log::error("%s", e.what());
        }
    });
    http_server_.listen(conf_.host, conf_.port);
}

int llama_lead::generate(const llama_tokens & tokens_list, size_t n_reuse)
{
    log::info("reusing %zu tokens.", n_reuse);
    llama_batch & batch = query_ctx_.batch; 

    auto encode_started_us = ggml_time_us();
    // evaluate the initial prompt
    auto bsz = conf_.n_batch;
    for (size_t i = n_reuse; i < tokens_list.size();)
    {
        llama_batch_clear(batch);
        size_t j;
        for (j = 0; j < bsz && i + j < tokens_list.size(); j++)
        {
            llama_batch_add(batch, tokens_list[i + j], i + j, { 0 }, false);
        }
        if (i + j == tokens_list.size())
        {
            batch.logits[batch.n_tokens - 1] = true;
        }
        if (llama_decode(llama_ctx_, batch) != 0)
        {
            log::error("llama_decode() failed");
            return 1;
        }
        i += j;
    }
    double encode_dur_s = (ggml_time_us() - encode_started_us) / 1000000.0;
    size_t n_encoded    = tokens_list.size() - n_reuse;
    log::info(
        "encoded %4zu tokens in %8.3f seconds, speed: %8.3f t/s",
        n_encoded,
        encode_dur_s,
        n_encoded / encode_dur_s
    );

    // how many tokens are currently accepted
    size_t n_cur  = tokens_list.size();

    llama_tokens input_seq, next_tokens;
    input_seq.push_back(tokens_list.back());

    int logits_from = batch.n_tokens - 1;
    int logits_to   = batch.n_tokens;
    const auto t_start = ggml_time_us();
    while (n_cur < query_ctx_.n_len)
    {
        next_tokens = greedy_tokens(model_, llama_ctx_, logits_from, logits_to);
        if (next_tokens.size() != input_seq.size())
        {
            log::error("invalid next tokens");
            return 1;
        }

        // this is where next_tokens start
        size_t next_tokens_pos = n_cur;
        // we always accept at least one new token
        n_cur += 1;
        for (size_t i = 0; i + 1 < input_seq.size(); i++)
        {
            if (next_tokens[i] == input_seq[i + 1])
            {
                n_cur += 1;
            }
            else
            {
                // reject. next_tokens[i] is the last correct one.
                next_tokens.erase(next_tokens.begin() + i + 1, next_tokens.end());
                break;
            }
        }

        // empty the non-matching portion of kv cache. 
        // n_cur is incremented at least once and will be > 0
        llama_kv_cache_seq_rm(llama_ctx_, 0, n_cur - 1, -1);

        bool done = false;
        for (size_t i = 0; i < next_tokens.size(); i++)
        {
            // TODO: what should we do here, is this correct
            if (next_tokens[i] == llama_token_eos(model_) || llama_token_is_eog(model_, next_tokens[i]))
            {
                done = true;
                next_tokens.erase(next_tokens.begin() + i, next_tokens.end());
                break;
            }
        }
        // append next_tokens to the output
        query_ctx_.output.insert(query_ctx_.output.end(), next_tokens.begin(), next_tokens.end());

        if (n_cur >= query_ctx_.n_len || done)
        {
            break;
        }

        // reconcile main and speculative
        {
            std::lock_guard<std::mutex> _lock(spec_ctx_.mtx);
            auto & spec = spec_ctx_.candidate;
            size_t n_match = 0;
            for (size_t i = 0; i < next_tokens.size() && i + next_tokens_pos < spec.size(); i++)
            {
                if (next_tokens[i] == spec[i + next_tokens_pos])
                {
                    n_match++;
                }
                else
                {
                    break;
                }
            }

            // Write accepted/rejected/not matched
            // this is slow and inefficient but for short strings doesn't matter 
            if (conf_.print_mode != "none")
            {
                std::string accepted = "";
                for (size_t i = next_tokens_pos; i < next_tokens_pos + n_match; i++)
                {
                    accepted += llama_token_to_piece(llama_ctx_, spec[i]);
                }
                dbg_accepted(accepted);
            }
            if (n_match != next_tokens.size())
            {
                if (conf_.print_mode == "all")
                {
                    std::string rejected = "";
                    for (size_t i = next_tokens_pos + n_match; i < spec.size(); i++)
                    {
                        rejected += llama_token_to_piece(llama_ctx_, spec[i]);
                    }
                    dbg_rejected(rejected);
                }
                if (conf_.print_mode != "none")
                {
                    std::string not_matched = "";
                    for (size_t i = n_match; i < next_tokens.size(); i++)
                    {
                        not_matched += llama_token_to_piece(llama_ctx_, next_tokens[i]);
                    }
                    dbg_not_matched(not_matched);
                }
            }

            // remove non-matched tokens
            if (n_match != next_tokens.size())
            {
                spec.erase(spec.begin() + next_tokens_pos, spec.end());
                for (const auto tok: next_tokens)
                {
                    spec.push_back(tok);
                }
            }
            spec_ctx_.n_approved     = next_tokens_pos + next_tokens.size();
            spec_ctx_.crc32_approved = crc32(spec.begin(), spec.begin() + next_tokens_pos + next_tokens.size());
            input_seq.assign(spec.begin() + n_cur - 1, spec.end());
        }

        llama_batch_clear(batch);
        if (input_seq.size() + n_cur > query_ctx_.n_len)
        {
            input_seq.resize(query_ctx_.n_len - n_cur);
        }
        // in some cases this might be not the most efficient thing to do.
        // for correctness just make the input size <= batch size
        if (input_seq.size() > bsz)
        {
            log::warn("trimming speculation to fit in batch size");
            input_seq.resize(bsz);
        }
        for (size_t i = 0; i < input_seq.size(); i++)
        {
            llama_batch_add(batch, input_seq[i], n_cur - 1 + i, { 0 }, true);
        }
        if (llama_decode(llama_ctx_, batch))
        {
            log::error("llama_decode() failed");
            return 1;
        }
        logits_from = 0;
        logits_to = input_seq.size();
    }

    if (conf_.print_mode != "none")
    {
        for (size_t i = 0; i < next_tokens.size(); i++)
        {
            auto sp = llama_token_to_piece(llama_ctx_, next_tokens[i]);
            dbg_not_matched(sp);
        }
    }
    double decode_dur_s = (ggml_time_us() - t_start) / 1000000.0;
    size_t n_decoded    = n_cur - tokens_list.size();
    log::info("decoded %4zu tokens in %8.3f seconds, speed: %8.3f t/s", n_decoded, decode_dur_s, n_decoded / decode_dur_s);

    return 0;
}

} // namespace llama_duo

int main(int argc, char ** argv)
{
    llama_backend_init();
    auto conf = llama_duo::gen_config(argc, argv);

    auto node = llama_duo::llama_lead::create(conf);
    if (node != nullptr)
    {
        node->serve();
    }
    llama_backend_free();

    return 0;
}
