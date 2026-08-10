#pragma once

// Byte-level BPE for GLM-5.2: GGUF vocab + merges, the GLM-4 pre-tokenizer,
// encode and decode.
//
// Tier note (see lib/README.md): the BPE machinery is generic to any
// `tokenizer.ggml.model == "gpt2"` vocab, but `pretok_split` implements the
// one regex GLM-4 declares. A model with a different `tokenizer.ggml.pre`
// needs a different splitter and nothing else.
//
// Why this is written out rather than borrowed: nano-glm links ggml, not
// llama, and llama.cpp's tokenizer comes with its vocab abstraction, its
// unicode tables and its own notion of what a "special" token is. What is
// borrowed — deliberately, because guessing would be silently wrong — is the
// \p{L} / \p{N} classification, generated from llama.cpp's tables into
// unicode_ranges.h so the two cannot drift apart on a Unicode revision.
//
// Correctness is measured, not asserted: `python tokenizer_check.py` compares
// our ids against llama-tokenize over a corpus and prints the disagreement
// rate. Treat that number as the contract.

#include "nano_model.h"
#include "unicode_ranges.h"

#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// character classes

static bool cp_in(const uint32_t (*ranges)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if      (cp <  ranges[mid][0]) hi = mid;
        else if (cp >  ranges[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

static bool cp_is_letter(uint32_t cp) {
    return cp_in(NANO_UNICODE_LETTER, sizeof(NANO_UNICODE_LETTER) / sizeof(*NANO_UNICODE_LETTER), cp);
}
static bool cp_is_number(uint32_t cp) {
    return cp_in(NANO_UNICODE_NUMBER, sizeof(NANO_UNICODE_NUMBER) / sizeof(*NANO_UNICODE_NUMBER), cp);
}
static bool cp_is_space(uint32_t cp) {
    for (uint32_t w : NANO_UNICODE_WHITESPACE) if (w == cp) return true;
    return false;
}
static bool cp_is_nl(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// ---------------------------------------------------------------------------
// UTF-8

static std::vector<uint32_t> utf8_to_cpts(const std::string & s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i];
        uint32_t cp;
        int n;
        if      ((c & 0x80) == 0x00) { cp = c;        n = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; n = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; n = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; n = 4; }
        else                         { cp = 0xFFFD;   n = 1; }   // stray continuation
        if (i + n > s.size()) { cp = 0xFFFD; n = 1; }
        for (int k = 1; k < n; k++) cp = (cp << 6) | (s[i + k] & 0x3F);
        out.push_back(cp);
        i += n;
    }
    return out;
}

static void cpt_to_utf8(uint32_t cp, std::string & out) {
    if (cp < 0x80) {
        out += (char) cp;
    } else if (cp < 0x800) {
        out += (char) (0xC0 | (cp >> 6));
        out += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char) (0xE0 | (cp >> 12));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    } else {
        out += (char) (0xF0 | (cp >> 18));
        out += (char) (0x80 | ((cp >> 12) & 0x3F));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    }
}

// ---------------------------------------------------------------------------
// GPT-2 byte alphabet
//
// Byte-level BPE runs over printable code points, not raw bytes, so that a
// merge table can be plain text. Bytes that are already printable map to
// themselves; the other 68 map to U+0100.. in order. This is the mapping that
// turns a space into "G-with-dot" and a newline into "C-with-dot" in the
// vocab dump, and reversing it is all `detokenize` does.

struct byte_alphabet {
    uint32_t to_cp[256];
    std::unordered_map<uint32_t, uint8_t> to_byte;

    byte_alphabet() {
        bool printable[256] = { false };
        for (int b = '!';    b <= '~';    b++) printable[b] = true;
        for (int b = 0xA1;   b <= 0xAC;   b++) printable[b] = true;
        for (int b = 0xAE;   b <= 0xFF;   b++) printable[b] = true;
        uint32_t next = 256;
        for (int b = 0; b < 256; b++) {
            to_cp[b] = printable[b] ? (uint32_t) b : next++;
            to_byte[to_cp[b]] = (uint8_t) b;
        }
    }
};

static const byte_alphabet & alphabet() {
    static const byte_alphabet a;
    return a;
}

// ---------------------------------------------------------------------------
// the GLM-4 pre-tokenizer
//
// llama.cpp declares it as one regex (llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_
// CHATGLM4):
//
//   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}{1,3}
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
//
// Hand-scanned rather than handed to std::regex, which has no Unicode
// property support at all. Alternation is ordered: at each position the first
// alternative that matches wins, and each is greedy within itself. Two need
// care because they lean on backtracking:
//
//   \s*[\r\n]+   greedy \s* backs off until the next code point is a newline,
//                so the match ends at the LAST newline in the whitespace run.
//   \s+(?!\S)    \s+ backs off by one when a non-space follows, which is what
//                leaves a single space to be picked up by the `[^\r\n\p{L}
//                \p{N}]?` prefix of the next word — the classic GPT-2 " word"
//                behaviour.

static std::vector<std::pair<size_t, size_t>> pretok_split(const std::vector<uint32_t> & cp) {
    std::vector<std::pair<size_t, size_t>> out;
    const size_t n = cp.size();
    size_t i = 0;

    auto lower = [](uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };

    while (i < n) {
        size_t j = i;

        // 1. contractions
        if (cp[i] == '\'' && i + 1 < n) {
            const uint32_t a = lower(cp[i + 1]);
            const uint32_t b = i + 2 < n ? lower(cp[i + 2]) : 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') {
                out.push_back({ i, i + 2 });
                i += 2;
                continue;
            }
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
                out.push_back({ i, i + 3 });
                i += 3;
                continue;
            }
        }

        // 2. [^\r\n\p{L}\p{N}]? \p{L}+
        {
            size_t k = i;
            if (!cp_is_nl(cp[k]) && !cp_is_letter(cp[k]) && !cp_is_number(cp[k])) k++;
            size_t s = k;
            while (k < n && cp_is_letter(cp[k])) k++;
            if (k > s) { out.push_back({ i, k }); i = k; continue; }
        }

        // 3. \p{N}{1,3}
        if (cp_is_number(cp[i])) {
            size_t k = i;
            while (k < n && k < i + 3 && cp_is_number(cp[k])) k++;
            out.push_back({ i, k });
            i = k;
            continue;
        }

        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t k = i;
            if (cp[k] == ' ') k++;
            size_t s = k;
            while (k < n && !cp_is_space(cp[k]) && !cp_is_letter(cp[k]) && !cp_is_number(cp[k])) k++;
            if (k > s) {
                while (k < n && cp_is_nl(cp[k])) k++;
                out.push_back({ i, k });
                i = k;
                continue;
            }
        }

        // 5. \s*[\r\n]+   (ends at the last newline in the whitespace run)
        if (cp_is_space(cp[i])) {
            size_t run = i;
            while (run < n && cp_is_space(cp[run])) run++;
            size_t last_nl = run;
            for (size_t k = run; k > i; k--) {
                if (cp_is_nl(cp[k - 1])) { last_nl = k; break; }
            }
            if (last_nl != run || cp_is_nl(cp[run - 1])) {
                if (last_nl > i) { out.push_back({ i, last_nl }); i = last_nl; continue; }
            }

            // 6. \s+(?!\S) — the whole run at end of text, else all but one
            const size_t end = (run == n) ? run : run - 1;
            if (end > i) { out.push_back({ i, end }); i = end; continue; }

            // 7. \s+
            out.push_back({ i, run });
            i = run;
            continue;
        }

        // nothing matched (should not happen): emit one code point and advance
        out.push_back({ i, i + 1 });
        i = j + 1;
    }
    return out;
}

// ---------------------------------------------------------------------------
// vocab

struct nano_vocab {
    std::vector<std::string> tokens;
    std::vector<int32_t>     types;                     // 1 normal, 3 control, 4 user-defined
    std::unordered_map<std::string, int32_t> id_of;
    std::unordered_map<std::string, int32_t> merge_rank; // "left right" -> rank

    int32_t eos_id = -1, eot_id = -1, bos_id = -1;

    int32_t id(const std::string & text) const {
        auto it = id_of.find(text);
        return it == id_of.end() ? -1 : it->second;
    }
    // Aborts rather than returning -1: a missing control token means the chat
    // template cannot be built, and continuing would produce a prompt that is
    // wrong in a way the model answers fluently.
    int32_t must(const std::string & text) const {
        const int32_t t = id(text);
        if (t < 0) NANO_ABORT("vocab has no token %s", text.c_str());
        return t;
    }
};

static void load_vocab(nano_vocab & V, const std::string & first_shard) {
    gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(first_shard.c_str(), gp);
    if (!g) NANO_ABORT("failed to read GGUF '%s'", first_shard.c_str());

    const std::string model = kv_str_opt(g, "tokenizer.ggml.model", "?");
    const std::string pre   = kv_str_opt(g, "tokenizer.ggml.pre", "?");
    if (model != "gpt2") {
        NANO_ABORT("tokenizer.ggml.model is '%s', only byte-level BPE ('gpt2') is ported",
                   model.c_str());
    }
    if (pre != "glm4") {
        NANO_ABORT("tokenizer.ggml.pre is '%s', only 'glm4' pre-tokenization is ported "
                   "(see pretok_split in lib/vocab.h)", pre.c_str());
    }

    const int64_t i_tok = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (i_tok < 0) NANO_ABORT("GGUF has no tokenizer.ggml.tokens");
    const size_t n_tok = gguf_get_arr_n(g, i_tok);
    V.tokens.resize(n_tok);
    V.id_of.reserve(n_tok * 2);
    for (size_t i = 0; i < n_tok; i++) {
        V.tokens[i] = gguf_get_arr_str(g, i_tok, i);
        V.id_of[V.tokens[i]] = (int32_t) i;
    }

    V.types.assign(n_tok, 1);
    const int64_t i_type = gguf_find_key(g, "tokenizer.ggml.token_type");
    if (i_type >= 0) {
        const int32_t * t = (const int32_t *) gguf_get_arr_data(g, i_type);
        for (size_t i = 0; i < n_tok && i < gguf_get_arr_n(g, i_type); i++) V.types[i] = t[i];
    }

    const int64_t i_mrg = gguf_find_key(g, "tokenizer.ggml.merges");
    if (i_mrg < 0) NANO_ABORT("GGUF has no tokenizer.ggml.merges");
    const size_t n_mrg = gguf_get_arr_n(g, i_mrg);
    V.merge_rank.reserve(n_mrg * 2);
    for (size_t i = 0; i < n_mrg; i++) {
        V.merge_rank[gguf_get_arr_str(g, i_mrg, i)] = (int32_t) i;
    }

    auto id_kv = [&](const char * key) -> int32_t {
        const int64_t k = gguf_find_key(g, key);
        return k < 0 ? -1 : (int32_t) kv_u32(g, key);
    };
    V.eos_id = id_kv("tokenizer.ggml.eos_token_id");
    V.eot_id = id_kv("tokenizer.ggml.eot_token_id");
    V.bos_id = id_kv("tokenizer.ggml.bos_token_id");

    gguf_free(g);
}

// ---------------------------------------------------------------------------
// encode / decode

// One pre-token, already mapped into the byte alphabet, reduced by merges.
// Lowest rank first, leftmost on a tie — the reference GPT-2 order.
static void bpe_merge(const nano_vocab & V, std::vector<std::string> & sym) {
    for (;;) {
        int32_t best_rank = -1;
        size_t  best_i    = 0;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            auto it = V.merge_rank.find(sym[i] + " " + sym[i + 1]);
            if (it != V.merge_rank.end() && (best_rank < 0 || it->second < best_rank)) {
                best_rank = it->second;
                best_i    = i;
            }
        }
        if (best_rank < 0) break;
        sym[best_i] += sym[best_i + 1];
        sym.erase(sym.begin() + best_i + 1);
    }
}

// Plain text -> ids. Control tokens in the text are NOT interpreted: the
// caller emits those itself (see chat_glm.h). That is deliberate — it means a
// user pasting "<|assistant|>" gets the literal characters rather than seizing
// control of the prompt.
static std::vector<int32_t> tokenize(const nano_vocab & V, const std::string & text) {
    const byte_alphabet & A = alphabet();
    const std::vector<uint32_t> cp = utf8_to_cpts(text);

    std::vector<int32_t> out;
    for (const auto & span : pretok_split(cp)) {
        // the span's original bytes, re-encoded into the byte alphabet
        std::string raw;
        for (size_t k = span.first; k < span.second; k++) cpt_to_utf8(cp[k], raw);

        std::vector<std::string> sym;
        sym.reserve(raw.size());
        for (unsigned char b : raw) {
            std::string s;
            cpt_to_utf8(A.to_cp[b], s);
            sym.push_back(s);
        }

        bpe_merge(V, sym);

        for (const std::string & s : sym) {
            const int32_t id = V.id(s);
            if (id < 0) NANO_ABORT("BPE produced a symbol not in the vocab: '%s'", s.c_str());
            out.push_back(id);
        }
    }
    return out;
}

// id -> the bytes it stands for. Control tokens have no byte content of their
// own; their spelling is returned as-is so a trace stays readable.
static std::string detokenize(const nano_vocab & V, int32_t id) {
    if (id < 0 || (size_t) id >= V.tokens.size()) return "";
    const std::string & t = V.tokens[id];
    if (V.types[id] == 3 || V.types[id] == 4) return t;   // control / user-defined

    const byte_alphabet & A = alphabet();
    std::string out;
    for (uint32_t c : utf8_to_cpts(t)) {
        auto it = A.to_byte.find(c);
        if (it != A.to_byte.end()) out += (char) it->second;
        else                       cpt_to_utf8(c, out);
    }
    return out;
}
