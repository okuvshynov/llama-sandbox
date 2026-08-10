#pragma once

// GLM-5.2's chat format, single turn.
//
// The GGUF ships a Jinja template (`tokenizer.chat_template`, ~120 lines of
// macros for tools, multimodal placeholders and multi-turn reasoning). Nothing
// here evaluates Jinja. What it does is reproduce the one path that a
// single-turn CLI takes, which after whitespace control collapses to:
//
//   [gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{prompt}<|assistant|><think>
//
// Read off the template with defaults: `enable_thinking` undefined -> true,
// `reasoning_effort` undefined -> 'max' (capitalized to "Max"), no tools, and
// `add_generation_prompt` -> the trailing `<|assistant|><think>`. With
// thinking off the tail is `<|assistant|><think></think>`, which is how the
// template tells the model to skip reasoning — not an empty string.
//
// Tier note (lib/README.md): this is the most model-specific file here. A
// second model brings its own; there is nothing to share but the idea.
//
// Control tokens are emitted as ids directly rather than spelled into the text
// and re-parsed. That is what keeps a prompt containing the literal characters
// "<|assistant|>" from becoming a turn boundary.

#include "vocab.h"

#include <string>
#include <vector>

struct glm_chat_ids {
    int32_t gmask, sop, system, user, assistant, think_open, think_close;
};

static glm_chat_ids glm_chat_lookup(const nano_vocab & V) {
    glm_chat_ids c;
    c.gmask       = V.must("[gMASK]");
    c.sop         = V.must("<sop>");
    c.system      = V.must("<|system|>");
    c.user        = V.must("<|user|>");
    c.assistant   = V.must("<|assistant|>");
    c.think_open  = V.must("<think>");
    c.think_close = V.must("</think>");
    return c;
}

// system may be empty. think=false appends the closed <think></think> pair.
static std::vector<int32_t> glm_chat_prompt(const nano_vocab & V,
                                            const std::string & user_text,
                                            const std::string & system_text,
                                            bool think) {
    const glm_chat_ids c = glm_chat_lookup(V);
    std::vector<int32_t> ids;

    auto text = [&](const std::string & s) {
        for (int32_t t : tokenize(V, s)) ids.push_back(t);
    };

    ids.push_back(c.gmask);
    ids.push_back(c.sop);

    // The reasoning-effort block the template emits before anything else.
    ids.push_back(c.system);
    text("Reasoning Effort: Max");

    if (!system_text.empty()) {
        ids.push_back(c.system);
        text(system_text);
    }

    ids.push_back(c.user);
    text(user_text);

    ids.push_back(c.assistant);
    ids.push_back(c.think_open);
    if (!think) ids.push_back(c.think_close);

    return ids;
}
