#include "logits_io.h"

#include <cstdio>
#include <cstring>

// --- v3 writer ---

bool qmlog_write(const std::string & path, const qmlog_file & f) {
    FILE * fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "qmlog_write: cannot open '%s'\n", path.c_str());
        return false;
    }

    // global header (72 bytes)
    fwrite(QMLOG_MAGIC, 1, 8, fp);

    uint32_t version = QMLOG_VERSION;
    fwrite(&version,      4, 1, fp);
    fwrite(&f.n_vocab,    4, 1, fp);
    fwrite(&f.n_prompts,  4, 1, fp);

    int32_t unused = 0;
    fwrite(&unused,       4, 1, fp);

    fwrite(&f.temp,       4, 1, fp);
    fwrite(&f.top_p,      4, 1, fp);
    fwrite(&f.top_k,      4, 1, fp);
    fwrite(&f.seed,       4, 1, fp);

    char reserved[28] = {};
    fwrite(reserved, 1, 28, fp);

    // prompt sections
    for (const auto & p : f.prompts) {
        // v3: path string
        int32_t path_len = (int32_t)p.path.size();
        fwrite(&path_len, 4, 1, fp);
        if (path_len > 0) {
            fwrite(p.path.data(), 1, path_len, fp);
        }

        fwrite(&p.n_tokens, 4, 1, fp);
        fwrite(&p.n_prompt, 4, 1, fp);
        fwrite(p.tokens.data(), 4, p.n_tokens, fp);

        const size_t n_logit_floats = (size_t)(p.n_tokens - 1) * f.n_vocab;
        fwrite(p.logits.data(), 4, n_logit_floats, fp);
    }

    fclose(fp);
    return true;
}

// --- header reading ---

struct raw_header {
    uint32_t version;
    int32_t  n_vocab;
    int32_t  field_16;  // n_tokens (v1) or n_prompts (v2/v3)
    int32_t  field_20;  // n_prompt (v1) or unused (v2/v3)
    float    temp;
    float    top_p;
    int32_t  top_k;
    uint32_t seed;
};

static bool read_global_header(FILE * fp, raw_header & rh) {
    char magic[8];
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, QMLOG_MAGIC, 8) != 0) {
        fprintf(stderr, "qmlog_read: bad magic\n");
        return false;
    }
    if (fread(&rh.version,  4, 1, fp) != 1) return false;
    if (rh.version < 1 || rh.version > 3) {
        fprintf(stderr, "qmlog_read: unsupported version %u\n", rh.version);
        return false;
    }
    if (fread(&rh.n_vocab,  4, 1, fp) != 1) return false;
    if (fread(&rh.field_16, 4, 1, fp) != 1) return false;
    if (fread(&rh.field_20, 4, 1, fp) != 1) return false;
    if (fread(&rh.temp,     4, 1, fp) != 1) return false;
    if (fread(&rh.top_p,    4, 1, fp) != 1) return false;
    if (fread(&rh.top_k,    4, 1, fp) != 1) return false;
    if (fread(&rh.seed,     4, 1, fp) != 1) return false;

    int32_t skip = (rh.version == 1) ? 32 : 28;
    if (fseek(fp, skip, SEEK_CUR) != 0) return false;

    return true;
}

static void fill_global(qmlog_file & f, const raw_header & rh) {
    f.version = rh.version;
    f.n_vocab = rh.n_vocab;
    f.temp    = rh.temp;
    f.top_p   = rh.top_p;
    f.top_k   = rh.top_k;
    f.seed    = rh.seed;
}

// --- v1 compat readers ---

static bool read_v1_full(FILE * fp, const raw_header & rh, qmlog_file & f) {
    fill_global(f, rh);
    f.n_prompts = 1;
    f.prompts.resize(1);

    auto & p = f.prompts[0];
    p.n_tokens = rh.field_16;
    p.n_prompt = rh.field_20;

    p.tokens.resize(p.n_tokens);
    if ((int32_t)fread(p.tokens.data(), 4, p.n_tokens, fp) != p.n_tokens) {
        fprintf(stderr, "qmlog_read: truncated tokens\n");
        return false;
    }

    const size_t n_logit_floats = (size_t)(p.n_tokens - 1) * rh.n_vocab;
    p.logits.resize(n_logit_floats);
    if (fread(p.logits.data(), 4, n_logit_floats, fp) != n_logit_floats) {
        fprintf(stderr, "qmlog_read: truncated logits\n");
        return false;
    }
    return true;
}

static bool read_v1_tokens(FILE * fp, const raw_header & rh, qmlog_file & f) {
    fill_global(f, rh);
    f.n_prompts = 1;
    f.prompts.resize(1);

    auto & p = f.prompts[0];
    p.n_tokens = rh.field_16;
    p.n_prompt = rh.field_20;

    p.tokens.resize(p.n_tokens);
    if ((int32_t)fread(p.tokens.data(), 4, p.n_tokens, fp) != p.n_tokens) {
        fprintf(stderr, "qmlog_read_tokens: truncated tokens\n");
        return false;
    }
    return true;
}

// --- v2 readers (no path field) ---

static bool read_v2_full(FILE * fp, const raw_header & rh, qmlog_file & f) {
    fill_global(f, rh);
    f.n_prompts = rh.field_16;
    f.prompts.resize(f.n_prompts);

    for (int32_t i = 0; i < f.n_prompts; i++) {
        auto & p = f.prompts[i];
        if (fread(&p.n_tokens, 4, 1, fp) != 1) return false;
        if (fread(&p.n_prompt, 4, 1, fp) != 1) return false;

        p.tokens.resize(p.n_tokens);
        if ((int32_t)fread(p.tokens.data(), 4, p.n_tokens, fp) != p.n_tokens) {
            fprintf(stderr, "qmlog_read: truncated tokens in prompt %d\n", i);
            return false;
        }

        const size_t n_logit_floats = (size_t)(p.n_tokens - 1) * rh.n_vocab;
        p.logits.resize(n_logit_floats);
        if (fread(p.logits.data(), 4, n_logit_floats, fp) != n_logit_floats) {
            fprintf(stderr, "qmlog_read: truncated logits in prompt %d\n", i);
            return false;
        }
    }
    return true;
}

static bool read_v2_tokens(FILE * fp, const raw_header & rh, qmlog_file & f) {
    fill_global(f, rh);
    f.n_prompts = rh.field_16;
    f.prompts.resize(f.n_prompts);

    for (int32_t i = 0; i < f.n_prompts; i++) {
        auto & p = f.prompts[i];
        if (fread(&p.n_tokens, 4, 1, fp) != 1) return false;
        if (fread(&p.n_prompt, 4, 1, fp) != 1) return false;

        p.tokens.resize(p.n_tokens);
        if ((int32_t)fread(p.tokens.data(), 4, p.n_tokens, fp) != p.n_tokens) {
            fprintf(stderr, "qmlog_read_tokens: truncated tokens in prompt %d\n", i);
            return false;
        }

        const size_t logit_bytes = (size_t)(p.n_tokens - 1) * rh.n_vocab * 4;
        if (fseek(fp, (long)logit_bytes, SEEK_CUR) != 0) return false;
    }
    return true;
}

// --- v3 readers (with path field) ---

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

static bool read_v3_full(FILE * fp, const raw_header & rh, qmlog_file & f) {
    fill_global(f, rh);
    f.n_prompts = rh.field_16;
    f.prompts.resize(f.n_prompts);

    for (int32_t i = 0; i < f.n_prompts; i++) {
        auto & p = f.prompts[i];
        if (!read_path(fp, p.path)) return false;
        if (fread(&p.n_tokens, 4, 1, fp) != 1) return false;
        if (fread(&p.n_prompt, 4, 1, fp) != 1) return false;

        p.tokens.resize(p.n_tokens);
        if ((int32_t)fread(p.tokens.data(), 4, p.n_tokens, fp) != p.n_tokens) {
            fprintf(stderr, "qmlog_read: truncated tokens in prompt %d\n", i);
            return false;
        }

        const size_t n_logit_floats = (size_t)(p.n_tokens - 1) * rh.n_vocab;
        p.logits.resize(n_logit_floats);
        if (fread(p.logits.data(), 4, n_logit_floats, fp) != n_logit_floats) {
            fprintf(stderr, "qmlog_read: truncated logits in prompt %d\n", i);
            return false;
        }
    }
    return true;
}

static bool read_v3_tokens(FILE * fp, const raw_header & rh, qmlog_file & f) {
    fill_global(f, rh);
    f.n_prompts = rh.field_16;
    f.prompts.resize(f.n_prompts);

    for (int32_t i = 0; i < f.n_prompts; i++) {
        auto & p = f.prompts[i];
        if (!read_path(fp, p.path)) return false;
        if (fread(&p.n_tokens, 4, 1, fp) != 1) return false;
        if (fread(&p.n_prompt, 4, 1, fp) != 1) return false;

        p.tokens.resize(p.n_tokens);
        if ((int32_t)fread(p.tokens.data(), 4, p.n_tokens, fp) != p.n_tokens) {
            fprintf(stderr, "qmlog_read_tokens: truncated tokens in prompt %d\n", i);
            return false;
        }

        const size_t logit_bytes = (size_t)(p.n_tokens - 1) * rh.n_vocab * 4;
        if (fseek(fp, (long)logit_bytes, SEEK_CUR) != 0) return false;
    }
    return true;
}

// --- public API ---

bool qmlog_read(const std::string & path, qmlog_file & f) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "qmlog_read: cannot open '%s'\n", path.c_str());
        return false;
    }

    raw_header rh;
    if (!read_global_header(fp, rh)) { fclose(fp); return false; }

    bool ok;
    if (rh.version == 1)      ok = read_v1_full(fp, rh, f);
    else if (rh.version == 2) ok = read_v2_full(fp, rh, f);
    else                      ok = read_v3_full(fp, rh, f);

    fclose(fp);
    return ok;
}

bool qmlog_read_tokens(const std::string & path, qmlog_file & f) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "qmlog_read_tokens: cannot open '%s'\n", path.c_str());
        return false;
    }

    raw_header rh;
    if (!read_global_header(fp, rh)) { fclose(fp); return false; }

    bool ok;
    if (rh.version == 1)      ok = read_v1_tokens(fp, rh, f);
    else if (rh.version == 2) ok = read_v2_tokens(fp, rh, f);
    else                      ok = read_v3_tokens(fp, rh, f);

    fclose(fp);
    return ok;
}
