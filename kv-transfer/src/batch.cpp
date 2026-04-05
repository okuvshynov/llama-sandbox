#include "batch.h"
#include "trace_io.h"
#include "stats_io.h"
#include "kl_utils.h"

#include "llama.h"
#include "common.h"
#include "chat.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

struct batch_params {
    std::string ref_model_path;
    std::string tgt_model_path;
    std::string tgt_tag;
    std::string prompts_dir;
    std::string output_dir;    // results base dir
    std::string ref_tag;
    int32_t     n_predict    = 2048;
    float       temp         = 0.6f;
    float       top_p        = 0.95f;
    int32_t     top_k        = 40;
    uint32_t    seed         = 42;
    int32_t     n_gpu_layers = 99;
    int32_t     n_ctx        = 0;  // 0 = auto
    int32_t     n_threads    = 0;
};

static bool parse_args(int argc, char ** argv, batch_params & params) {
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "-m-ref") == 0 && i + 1 < argc) {
            params.ref_model_path = argv[++i];
        } else if (strcmp(arg, "-m-tgt") == 0 && i + 1 < argc) {
            params.tgt_model_path = argv[++i];
        } else if (strcmp(arg, "--tgt-tag") == 0 && i + 1 < argc) {
            params.tgt_tag = argv[++i];
        } else if (strcmp(arg, "--ref-tag") == 0 && i + 1 < argc) {
            params.ref_tag = argv[++i];
        } else if (strcmp(arg, "--prompts") == 0 && i + 1 < argc) {
            params.prompts_dir = argv[++i];
        } else if (strcmp(arg, "-o") == 0 && i + 1 < argc) {
            params.output_dir = argv[++i];
        } else if (strcmp(arg, "-n") == 0 && i + 1 < argc) {
            params.n_predict = atoi(argv[++i]);
        } else if (strcmp(arg, "--temp") == 0 && i + 1 < argc) {
            params.temp = atof(argv[++i]);
        } else if (strcmp(arg, "--top-p") == 0 && i + 1 < argc) {
            params.top_p = atof(argv[++i]);
        } else if (strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
            params.top_k = atoi(argv[++i]);
        } else if (strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            params.seed = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(arg, "-ngl") == 0 && i + 1 < argc) {
            params.n_gpu_layers = atoi(argv[++i]);
        } else if (strcmp(arg, "-c") == 0 && i + 1 < argc) {
            params.n_ctx = atoi(argv[++i]);
        } else if (strcmp(arg, "-t") == 0 && i + 1 < argc) {
            params.n_threads = atoi(argv[++i]);
        } else {
            fprintf(stderr, "batch: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.ref_model_path.empty() || params.tgt_model_path.empty() ||
        params.tgt_tag.empty() || params.ref_tag.empty() ||
        params.prompts_dir.empty() || params.output_dir.empty()) {
        fprintf(stderr,
            "Usage: kv-transfer batch [options]\n"
            "  -m-ref <path>     reference model GGUF\n"
            "  -m-tgt <path>     target model GGUF\n"
            "  --ref-tag <tag>   reference tag (for output directory)\n"
            "  --tgt-tag <tag>   target tag (for output subdirectory)\n"
            "  --prompts <dir>   directory with prompt .txt files\n"
            "  -o <dir>          results output directory\n"
            "  -n <int>          tokens to generate (default: 2048)\n"
            "  --temp <f>        temperature (default: 0.6)\n"
            "  --top-p <f>       top-p (default: 0.95)\n"
            "  --top-k <int>     top-k (default: 40)\n"
            "  --seed <int>      RNG seed (default: 42)\n"
            "  -ngl <int>        GPU layers (default: 99)\n"
            "  -c <int>          context size (default: auto)\n"
            "  -t <int>          threads (default: llama.cpp default)\n");
        return false;
    }
    return true;
}

static std::vector<std::string> list_prompts(const std::string & dir) {
    std::vector<std::string> result;
    DIR * d = opendir(dir.c_str());
    if (!d) return result;
    struct dirent * ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name = ent->d_name;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".txt") {
            result.push_back(name.substr(0, name.size() - 4));
        }
    }
    closedir(d);
    std::sort(result.begin(), result.end());
    return result;
}

static std::string read_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) return "";
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s(sz, '\0');
    fread(&s[0], 1, sz, f);
    fclose(f);
    return s;
}

static void mkdir_p(const std::string & path) {
    std::string cmd = "mkdir -p '" + path + "'";
    system(cmd.c_str());
}

static bool file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "r");
    if (f) { fclose(f); return true; }
    return false;
}

// decode tokens in batches, optionally compute KL against ref logits
static bool batch_decode(
    llama_context * ctx, llama_batch & batch,
    const int32_t * tokens, int32_t n_tokens, int32_t n_prompt,
    int32_t n_batch, bool need_gen_logits,
    // if ref_logits non-null, compute KL inline
    const float * ref_logits, int32_t n_vocab, double temp,
    std::vector<float> * kl_out, std::vector<uint8_t> * top1_out,
    // if collect_logits, store raw logits (for ref phase)
    std::vector<float> * logits_out
) {
    std::vector<double> log_p_ref, log_p_tgt;

    int32_t n_processed = 0;
    while (n_processed < n_tokens) {
        common_batch_clear(batch);
        int32_t batch_end = std::min(n_tokens, n_processed + n_batch);
        for (int32_t i = n_processed; i < batch_end; i++) {
            bool need_logits = need_gen_logits && (i >= n_prompt);
            common_batch_add(batch, tokens[i], i, {0}, need_logits);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "batch_decode: failed at position %d\n", n_processed);
            return false;
        }
        for (int32_t i = n_processed; i < batch_end; i++) {
            if (i < n_prompt) continue;
            int32_t gen_idx = i - n_prompt;
            const float * logits = llama_get_logits_ith(ctx, i - n_processed);

            if (logits_out) {
                logits_out->insert(logits_out->end(), logits, logits + n_vocab);
            }
            if (ref_logits && kl_out && top1_out) {
                const float * rl = ref_logits + (size_t)gen_idx * n_vocab;
                log_softmax_temp(rl, n_vocab, temp, log_p_ref);
                log_softmax_temp(logits, n_vocab, temp, log_p_tgt);
                (*kl_out)[gen_idx] = (float)kl_divergence(log_p_ref, log_p_tgt, n_vocab);
                (*top1_out)[gen_idx] = (argmax(rl, n_vocab) == argmax(logits, n_vocab)) ? 1 : 0;
            }
        }
        n_processed = batch_end;
    }
    return true;
}

int cmd_batch(int argc, char ** argv) {
    batch_params params;
    if (!parse_args(argc, argv, params)) return 1;

    auto prompt_names = list_prompts(params.prompts_dir);
    if (prompt_names.empty()) {
        fprintf(stderr, "batch: no .txt files in '%s'\n", params.prompts_dir.c_str());
        return 1;
    }
    fprintf(stderr, "batch: found %zu prompts\n", prompt_names.size());

    // set up output directories
    std::string ref_dir = params.output_dir + "/" + params.ref_tag;
    std::string tgt_dir = ref_dir + "/" + params.tgt_tag;
    std::string log_dir = ref_dir + "/logs";
    mkdir_p(tgt_dir);
    mkdir_p(log_dir);

    // load models
    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    fprintf(stderr, "\n=== Loading ref model ===\n");
    llama_model * ref_model = llama_model_load_from_file(params.ref_model_path.c_str(), model_params);
    if (!ref_model) {
        fprintf(stderr, "batch: failed to load ref model\n");
        return 1;
    }

    fprintf(stderr, "\n=== Loading target model ===\n");
    llama_model * tgt_model = llama_model_load_from_file(params.tgt_model_path.c_str(), model_params);
    if (!tgt_model) {
        fprintf(stderr, "batch: failed to load target model\n");
        llama_model_free(ref_model);
        return 1;
    }

    const llama_vocab * ref_vocab = llama_model_get_vocab(ref_model);
    const int32_t n_vocab = llama_vocab_n_tokens(ref_vocab);

    const llama_vocab * tgt_vocab = llama_model_get_vocab(tgt_model);
    if (llama_vocab_n_tokens(tgt_vocab) != n_vocab) {
        fprintf(stderr, "batch: vocab mismatch: ref=%d, tgt=%d\n",
                n_vocab, llama_vocab_n_tokens(tgt_vocab));
        llama_model_free(tgt_model); llama_model_free(ref_model);
        return 1;
    }

    // chat template
    auto tmpls = common_chat_templates_init(ref_model, "");

    // sampler
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.seed));

    const double temp = (params.temp > 0.0f) ? (double)params.temp : 1.0;
    std::string state_path = ref_dir + "/.batch_kv_state.tmp";

    fprintf(stderr, "\n=== Processing %zu prompts ===\n\n", prompt_names.size());

    for (size_t pi = 0; pi < prompt_names.size(); pi++) {
        const std::string & name = prompt_names[pi];
        fprintf(stderr, "--- [%zu/%zu] %s ---\n", pi + 1, prompt_names.size(), name.c_str());

        std::string ref_bin_path  = ref_dir + "/" + name + "-ref.bin";
        std::string ref_txt_path  = ref_dir + "/" + name + "-ref.txt";
        std::string tgt_stats_path  = tgt_dir + "/" + name + "-target.bin";
        std::string hoff_stats_path = tgt_dir + "/" + name + "-handoff.bin";

        bool need_ref     = !file_exists(ref_bin_path);
        bool need_target  = !file_exists(tgt_stats_path);
        bool need_handoff = !file_exists(hoff_stats_path);

        if (!need_ref && !need_target && !need_handoff) {
            fprintf(stderr, "  all outputs exist, skipping\n\n");
            continue;
        }

        // we always need ref data (tokens + logits) for target/handoff
        trace_file ref_trace;
        int32_t n_prompt_tokens = 0;
        int32_t n_gen = 0;

        if (need_ref) {
            // --- Ref phase: generate tokens autoregressively ---
            fprintf(stderr, "  [ref] generating...\n");

            std::string prompt_text = read_file(params.prompts_dir + "/" + name + ".txt");
            std::string formatted = prompt_text;
            if (tmpls) {
                common_chat_templates_inputs inputs;
                inputs.messages = {{ "user", prompt_text }};
                inputs.add_generation_prompt = true;
                inputs.use_jinja = true;
                auto result = common_chat_templates_apply(tmpls.get(), inputs);
                if (!result.prompt.empty()) formatted = result.prompt;
            }

            std::vector<llama_token> prompt_tokens = common_tokenize(ref_vocab, formatted, true, true);
            n_prompt_tokens = (int32_t)prompt_tokens.size();
            int32_t total_tokens = n_prompt_tokens + params.n_predict;

            // create ref context sized for this prompt
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx   = params.n_ctx > 0 ? params.n_ctx : total_tokens;
            ctx_params.n_batch = std::max(params.n_ctx > 0 ? params.n_ctx : total_tokens, total_tokens);
            if (params.n_threads > 0) {
                ctx_params.n_threads       = params.n_threads;
                ctx_params.n_threads_batch = params.n_threads;
            }

            llama_context * ref_ctx = llama_init_from_model(ref_model, ctx_params);
            if (!ref_ctx) {
                fprintf(stderr, "  [ref] failed to create context, skipping\n\n");
                continue;
            }

            const int32_t n_batch = llama_n_batch(ref_ctx);
            llama_batch batch = llama_batch_init(n_batch, 0, 1);

            // decode prompt
            int32_t n_decoded = 0;
            while (n_decoded < n_prompt_tokens) {
                common_batch_clear(batch);
                int32_t batch_end = std::min(n_prompt_tokens, n_decoded + n_batch);
                for (int32_t i = n_decoded; i < batch_end; i++) {
                    common_batch_add(batch, prompt_tokens[i], i, {0}, i > 0);
                }
                if (llama_decode(ref_ctx, batch) != 0) {
                    fprintf(stderr, "  [ref] decode failed\n");
                    llama_batch_free(batch); llama_free(ref_ctx);
                    continue;
                }
                n_decoded = batch_end;
            }

            // save KV state (post-prompt, pre-generation) for handoff
            if (need_handoff) {
                llama_state_save_file(ref_ctx, state_path.c_str(),
                                      prompt_tokens.data(), n_prompt_tokens);
            }

            // generate autoregressively
            std::vector<int32_t> all_tokens(prompt_tokens.begin(), prompt_tokens.end());
            all_tokens.reserve(total_tokens);
            std::vector<float> all_logits;
            all_logits.reserve((size_t)params.n_predict * n_vocab);
            std::string generated_text;

            for (int32_t i = 0; i < params.n_predict; i++) {
                llama_token new_token = llama_sampler_sample(smpl, ref_ctx, -1);
                if (llama_vocab_is_eog(ref_vocab, new_token)) {
                    fprintf(stderr, "  [ref] EOS at step %d\n", i);
                    break;
                }
                all_tokens.push_back(new_token);
                generated_text += common_token_to_piece(ref_vocab, new_token);

                common_batch_clear(batch);
                common_batch_add(batch, new_token, n_prompt_tokens + i, {0}, true);
                if (llama_decode(ref_ctx, batch) != 0) break;

                const float * logits = llama_get_logits_ith(ref_ctx, 0);
                all_logits.insert(all_logits.end(), logits, logits + n_vocab);
            }

            n_gen = (int32_t)all_tokens.size() - n_prompt_tokens;
            fprintf(stderr, "  [ref] %d prompt + %d gen tokens\n", n_prompt_tokens, n_gen);

            // write ref.bin
            ref_trace.n_vocab   = n_vocab;
            ref_trace.n_prompts = 1;
            ref_trace.temp      = params.temp;
            ref_trace.top_p     = params.top_p;
            ref_trace.top_k     = params.top_k;
            ref_trace.seed      = params.seed;

            trace_entry & p = ref_trace.prompts.emplace_back();
            p.path     = name;
            p.n_tokens = (int32_t)all_tokens.size();
            p.n_prompt = n_prompt_tokens;
            p.tokens   = std::move(all_tokens);
            p.logits   = std::move(all_logits);

            trace_write(ref_bin_path, ref_trace);

            // write ref.txt
            FILE * txt_fp = fopen(ref_txt_path.c_str(), "w");
            if (txt_fp) { fprintf(txt_fp, "%s\n", generated_text.c_str()); fclose(txt_fp); }

            llama_perf_context_print(ref_ctx);
            llama_batch_free(batch);
            llama_free(ref_ctx);
        } else {
            // load existing ref.bin
            fprintf(stderr, "  [ref] loading existing %s\n", ref_bin_path.c_str());
            if (!trace_read(ref_bin_path, ref_trace)) {
                fprintf(stderr, "  [ref] failed to read, skipping\n\n");
                continue;
            }
            n_prompt_tokens = ref_trace.prompts[0].n_prompt;
            n_gen = ref_trace.prompts[0].n_tokens - n_prompt_tokens;
        }

        const auto & rp = ref_trace.prompts[0];
        const int32_t n_total = rp.n_tokens;

        // --- Target phase: batch decode all tokens, compute KL ---
        if (need_target) {
            fprintf(stderr, "  [target] batch eval...\n");

            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx   = params.n_ctx > 0 ? params.n_ctx : n_total;
            ctx_params.n_batch = n_total;
            if (params.n_threads > 0) {
                ctx_params.n_threads       = params.n_threads;
                ctx_params.n_threads_batch = params.n_threads;
            }

            llama_context * tgt_ctx = llama_init_from_model(tgt_model, ctx_params);
            if (!tgt_ctx) {
                fprintf(stderr, "  [target] failed to create context, skipping\n");
            } else {
                const int32_t n_batch = llama_n_batch(tgt_ctx);
                llama_batch batch = llama_batch_init(n_batch, 0, 1);

                std::vector<float>   kl(n_gen);
                std::vector<uint8_t> top1(n_gen);

                batch_decode(tgt_ctx, batch, rp.tokens.data(), n_total, n_prompt_tokens,
                             n_batch, true, rp.logits.data(), n_vocab, temp,
                             &kl, &top1, nullptr);

                stats_file sf;
                sf.n_gen    = n_gen;
                sf.n_prompt = n_prompt_tokens;
                sf.temp     = (float)temp;
                sf.kl         = std::move(kl);
                sf.top1_match = std::move(top1);
                stats_write(tgt_stats_path, sf);

                fprintf(stderr, "  [target] done (%d tokens)\n", n_gen);
                llama_perf_context_print(tgt_ctx);
                llama_batch_free(batch);
                llama_free(tgt_ctx);
            }
        } else {
            fprintf(stderr, "  [target] exists, skipping\n");
        }

        // --- Handoff phase: restore ref KV, batch decode gen tokens ---
        if (need_handoff) {
            fprintf(stderr, "  [handoff] batch eval with ref KV...\n");

            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx   = params.n_ctx > 0 ? params.n_ctx : n_total;
            ctx_params.n_batch = params.n_ctx > 0 ? params.n_ctx : n_total;
            if (params.n_threads > 0) {
                ctx_params.n_threads       = params.n_threads;
                ctx_params.n_threads_batch = params.n_threads;
            }

            // we need the ref KV state — if we just did the ref phase, state_path exists.
            // if ref was loaded from disk, we need to re-decode the prompt with ref model.
            if (!need_ref && !file_exists(state_path)) {
                fprintf(stderr, "  [handoff] re-decoding prompt with ref model for KV state...\n");
                llama_context * ref_ctx = llama_init_from_model(ref_model, ctx_params);
                if (ref_ctx) {
                    const int32_t n_batch = llama_n_batch(ref_ctx);
                    llama_batch batch = llama_batch_init(n_batch, 0, 1);
                    int32_t n_decoded = 0;
                    while (n_decoded < n_prompt_tokens) {
                        common_batch_clear(batch);
                        int32_t batch_end = std::min(n_prompt_tokens, n_decoded + n_batch);
                        for (int32_t i = n_decoded; i < batch_end; i++) {
                            common_batch_add(batch, rp.tokens[i], i, {0}, false);
                        }
                        llama_decode(ref_ctx, batch);
                        n_decoded = batch_end;
                    }
                    llama_state_save_file(ref_ctx, state_path.c_str(),
                                          rp.tokens.data(), n_prompt_tokens);
                    llama_perf_context_print(ref_ctx);
                    llama_batch_free(batch);
                    llama_free(ref_ctx);
                }
            }

            llama_context * tgt_ctx = llama_init_from_model(tgt_model, ctx_params);
            if (!tgt_ctx) {
                fprintf(stderr, "  [handoff] failed to create context, skipping\n");
            } else {
                // restore ref KV state
                std::vector<llama_token> loaded_tokens(n_prompt_tokens);
                size_t n_loaded = 0;
                bool ok = llama_state_load_file(tgt_ctx, state_path.c_str(),
                                                 loaded_tokens.data(), n_prompt_tokens, &n_loaded);
                if (!ok) {
                    fprintf(stderr, "  [handoff] failed to load KV state\n");
                    llama_free(tgt_ctx);
                } else {
                    const int32_t n_batch = llama_n_batch(tgt_ctx);
                    llama_batch batch = llama_batch_init(n_batch, 0, 1);

                    std::vector<float>   kl(n_gen);
                    std::vector<uint8_t> top1(n_gen);

                    // decode only generation tokens (KV cache has prompt from ref)
                    int32_t n_gen_processed = 0;
                    std::vector<double> log_p_ref, log_p_tgt;
                    while (n_gen_processed < n_gen) {
                        common_batch_clear(batch);
                        int32_t batch_end = std::min(n_gen, n_gen_processed + n_batch);
                        for (int32_t i = n_gen_processed; i < batch_end; i++) {
                            int32_t pos = n_prompt_tokens + i;
                            common_batch_add(batch, rp.tokens[pos], pos, {0}, true);
                        }
                        if (llama_decode(tgt_ctx, batch) != 0) {
                            fprintf(stderr, "  [handoff] decode failed at %d\n", n_gen_processed);
                            break;
                        }
                        for (int32_t i = n_gen_processed; i < batch_end; i++) {
                            const float * tgt_logits = llama_get_logits_ith(tgt_ctx, i - n_gen_processed);
                            const float * ref_logits = rp.logits.data() + (size_t)i * n_vocab;
                            log_softmax_temp(ref_logits, n_vocab, temp, log_p_ref);
                            log_softmax_temp(tgt_logits, n_vocab, temp, log_p_tgt);
                            kl[i] = (float)kl_divergence(log_p_ref, log_p_tgt, n_vocab);
                            top1[i] = (argmax(ref_logits, n_vocab) == argmax(tgt_logits, n_vocab)) ? 1 : 0;
                        }
                        n_gen_processed = batch_end;
                    }

                    stats_file sf;
                    sf.n_gen    = n_gen;
                    sf.n_prompt = n_prompt_tokens;
                    sf.temp     = (float)temp;
                    sf.kl         = std::move(kl);
                    sf.top1_match = std::move(top1);
                    stats_write(hoff_stats_path, sf);

                    fprintf(stderr, "  [handoff] done (%d tokens)\n", n_gen);
                    llama_perf_context_print(tgt_ctx);
                    llama_batch_free(batch);
                    llama_free(tgt_ctx);
                }
            }

            // clean up temp state file
            remove(state_path.c_str());
        } else {
            fprintf(stderr, "  [handoff] exists, skipping\n");
        }

        fprintf(stderr, "\n");
    }

    llama_sampler_free(smpl);
    llama_model_free(tgt_model);
    llama_model_free(ref_model);

    fprintf(stderr, "batch: done, processed %zu prompts\n", prompt_names.size());
    return 0;
}
