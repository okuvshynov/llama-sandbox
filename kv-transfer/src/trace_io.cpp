#include "trace_io.h"

#include <cstdio>
#include <cstring>

// --- v4 binary layout ---
//
// Global header (72 bytes):
//   [0..7]   magic "qmlogits"
//   [8..11]  version (uint32 = 4)
//   [12..15] n_vocab (int32)
//   [16..19] n_prompts (int32)
//   [20..23] reserved (int32, zero)
//   [24..27] temp (float)
//   [28..31] top_p (float)
//   [32..35] top_k (int32)
//   [36..39] seed (uint32)
//   [40..71] reserved (28 bytes, zero)
//
// Per-prompt section (repeated n_prompts times):
//   path_len (int32) + path bytes
//   n_tokens (int32)
//   n_prompt (int32)
//   tokens   (int32 * n_tokens)
//   logits   (float * (n_tokens - n_prompt) * n_vocab)  [generation only]

static constexpr size_t HEADER_SIZE = 72;

bool trace_write(const std::string & path, const trace_file & f) {
    FILE * fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "trace_write: cannot open '%s'\n", path.c_str());
        return false;
    }

    // global header
    fwrite(TRACE_MAGIC, 1, 8, fp);

    uint32_t version = TRACE_VERSION;
    fwrite(&version,      4, 1, fp);
    fwrite(&f.n_vocab,    4, 1, fp);
    fwrite(&f.n_prompts,  4, 1, fp);

    int32_t reserved_i32 = 0;
    fwrite(&reserved_i32, 4, 1, fp);

    fwrite(&f.temp,       4, 1, fp);
    fwrite(&f.top_p,      4, 1, fp);
    fwrite(&f.top_k,      4, 1, fp);
    fwrite(&f.seed,       4, 1, fp);

    char reserved[28] = {};
    fwrite(reserved, 1, 28, fp);

    // prompt sections
    for (const auto & p : f.prompts) {
        int32_t path_len = (int32_t)p.path.size();
        fwrite(&path_len, 4, 1, fp);
        if (path_len > 0) {
            fwrite(p.path.data(), 1, path_len, fp);
        }

        fwrite(&p.n_tokens, 4, 1, fp);
        fwrite(&p.n_prompt, 4, 1, fp);
        fwrite(p.tokens.data(), 4, p.n_tokens, fp);

        const size_t n_gen = p.n_tokens - p.n_prompt;
        const size_t n_logit_floats = n_gen * f.n_vocab;
        fwrite(p.logits.data(), 4, n_logit_floats, fp);
    }

    fclose(fp);
    return true;
}

// --- reader helpers ---

struct raw_header {
    int32_t  n_vocab;
    int32_t  n_prompts;
    float    temp;
    float    top_p;
    int32_t  top_k;
    uint32_t seed;
};

static bool read_global_header(FILE * fp, raw_header & rh) {
    char magic[8];
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, TRACE_MAGIC, 8) != 0) {
        fprintf(stderr, "trace_read: bad magic\n");
        return false;
    }

    uint32_t version;
    if (fread(&version, 4, 1, fp) != 1) return false;
    if (version != TRACE_VERSION) {
        fprintf(stderr, "trace_read: unsupported version %u (expected %u)\n", version, TRACE_VERSION);
        return false;
    }

    if (fread(&rh.n_vocab,    4, 1, fp) != 1) return false;
    if (fread(&rh.n_prompts,  4, 1, fp) != 1) return false;

    // skip reserved int32
    if (fseek(fp, 4, SEEK_CUR) != 0) return false;

    if (fread(&rh.temp,       4, 1, fp) != 1) return false;
    if (fread(&rh.top_p,      4, 1, fp) != 1) return false;
    if (fread(&rh.top_k,      4, 1, fp) != 1) return false;
    if (fread(&rh.seed,       4, 1, fp) != 1) return false;

    // skip 28 bytes reserved
    if (fseek(fp, 28, SEEK_CUR) != 0) return false;

    return true;
}

static bool read_path(FILE * fp, std::string & path) {
    int32_t path_len = 0;
    if (fread(&path_len, 4, 1, fp) != 1) return false;
    if (path_len > 0) {
        path.resize(path_len);
        if ((int32_t)fread(&path[0], 1, path_len, fp) != path_len) return false;
    } else {
        path.clear();
    }
    return true;
}

// --- public API ---

bool trace_read(const std::string & path, trace_file & f) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "trace_read: cannot open '%s'\n", path.c_str());
        return false;
    }

    raw_header rh;
    if (!read_global_header(fp, rh)) { fclose(fp); return false; }

    f.version    = TRACE_VERSION;
    f.n_vocab    = rh.n_vocab;
    f.n_prompts  = rh.n_prompts;
    f.temp       = rh.temp;
    f.top_p      = rh.top_p;
    f.top_k      = rh.top_k;
    f.seed       = rh.seed;
    f.prompts.resize(f.n_prompts);

    for (int32_t i = 0; i < f.n_prompts; i++) {
        auto & p = f.prompts[i];
        if (!read_path(fp, p.path)) { fclose(fp); return false; }
        if (fread(&p.n_tokens, 4, 1, fp) != 1) { fclose(fp); return false; }
        if (fread(&p.n_prompt, 4, 1, fp) != 1) { fclose(fp); return false; }

        p.tokens.resize(p.n_tokens);
        if ((int32_t)fread(p.tokens.data(), 4, p.n_tokens, fp) != p.n_tokens) {
            fprintf(stderr, "trace_read: truncated tokens in prompt %d\n", i);
            fclose(fp); return false;
        }

        const size_t n_gen = p.n_tokens - p.n_prompt;
        const size_t n_logit_floats = n_gen * rh.n_vocab;
        p.logits.resize(n_logit_floats);
        if (fread(p.logits.data(), 4, n_logit_floats, fp) != n_logit_floats) {
            fprintf(stderr, "trace_read: truncated logits in prompt %d\n", i);
            fclose(fp); return false;
        }
    }

    fclose(fp);
    return true;
}

bool trace_read_tokens(const std::string & path, trace_file & f) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "trace_read_tokens: cannot open '%s'\n", path.c_str());
        return false;
    }

    raw_header rh;
    if (!read_global_header(fp, rh)) { fclose(fp); return false; }

    f.version    = TRACE_VERSION;
    f.n_vocab    = rh.n_vocab;
    f.n_prompts  = rh.n_prompts;
    f.temp       = rh.temp;
    f.top_p      = rh.top_p;
    f.top_k      = rh.top_k;
    f.seed       = rh.seed;
    f.prompts.resize(f.n_prompts);

    for (int32_t i = 0; i < f.n_prompts; i++) {
        auto & p = f.prompts[i];
        if (!read_path(fp, p.path)) { fclose(fp); return false; }
        if (fread(&p.n_tokens, 4, 1, fp) != 1) { fclose(fp); return false; }
        if (fread(&p.n_prompt, 4, 1, fp) != 1) { fclose(fp); return false; }

        p.tokens.resize(p.n_tokens);
        if ((int32_t)fread(p.tokens.data(), 4, p.n_tokens, fp) != p.n_tokens) {
            fprintf(stderr, "trace_read_tokens: truncated tokens in prompt %d\n", i);
            fclose(fp); return false;
        }

        const size_t n_gen = p.n_tokens - p.n_prompt;
        const size_t logit_bytes = n_gen * rh.n_vocab * 4;
        if (fseek(fp, (long)logit_bytes, SEEK_CUR) != 0) { fclose(fp); return false; }
    }

    fclose(fp);
    return true;
}
