#pragma once

// Where an app gets prompt token *ids* from, when it is not in the business of
// turning text into them. Two sources, both exact:
//
//   -i <lkldtopk>   the prompt half of an existing logits file
//   -T <id,id,...>  a literal list
//
// No tokenizer here on purpose. nano-glm and nano-bench both take ids as their
// interface so that a run is reproducible from the command line alone;
// nano-chat is where text becomes ids, and its `--dry-run` prints a list that
// pastes straight into `-T`.

#include "gguf_store.h"

#include "logits_file.h"

#include <cstdlib>
#include <string>
#include <vector>

// Exactly one of input_bin / tokens_str should be set. `label` is filled with
// the sequence label when reading a file, so a report can say which prompt it
// is talking about.
static std::vector<int32_t> load_prompt_tokens(const std::string & input_bin,
                                               const std::string & tokens_str,
                                               std::string & label,
                                               const char * who) {
    std::vector<int32_t> toks;
    if (!input_bin.empty()) {
        lkld_file f;
        if (!lkld_read(input_bin, f)) NANO_ABORT("cannot read '%s'", input_bin.c_str());
        if (f.seqs.empty()) NANO_ABORT("'%s' has no sequences", input_bin.c_str());
        const lkld_seq & s = f.seqs[0];
        toks.assign(s.tokens.begin(), s.tokens.begin() + s.n_prompt);
        label = s.label;
        fprintf(stderr, "%s: prompt = %d tokens from %s (seq label '%s')\n",
                who, s.n_prompt, input_bin.c_str(), s.label.c_str());
    } else {
        const char * s = tokens_str.c_str();
        while (*s) {
            char * end = nullptr;
            long v = strtol(s, &end, 10);
            if (end == s) NANO_ABORT("bad token list near '%s'", s);
            toks.push_back((int32_t) v);
            s = *end == ',' ? end + 1 : end;
        }
        label = "tokens";
    }
    if (toks.empty()) NANO_ABORT("empty prompt");
    return toks;
}
