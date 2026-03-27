#include "logits_io.h"

#include <cstdio>
#include <cstring>

bool qmlog_write(const std::string & path, const qmlog_file & f) {
    FILE * fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "qmlog_write: cannot open '%s'\n", path.c_str());
        return false;
    }

    const auto & h = f.header;

    // magic (8 bytes)
    fwrite(QMLOG_MAGIC, 1, 8, fp);

    // header fields
    fwrite(&h.version,  4, 1, fp);
    fwrite(&h.n_vocab,  4, 1, fp);
    fwrite(&h.n_tokens, 4, 1, fp);
    fwrite(&h.n_prompt, 4, 1, fp);
    fwrite(&h.temp,     4, 1, fp);
    fwrite(&h.top_p,    4, 1, fp);
    fwrite(&h.top_k,    4, 1, fp);
    fwrite(&h.seed,     4, 1, fp);

    // reserved (32 bytes of zeros)
    char reserved[32] = {};
    fwrite(reserved, 1, 32, fp);

    // tokens
    fwrite(f.tokens.data(), 4, h.n_tokens, fp);

    // logits: (n_tokens - 1) * n_vocab floats
    const size_t n_logits = (size_t)(h.n_tokens - 1) * h.n_vocab;
    fwrite(f.logits.data(), 4, n_logits, fp);

    fclose(fp);
    return true;
}

static bool read_header(FILE * fp, qmlog_header & h) {
    char magic[8];
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, QMLOG_MAGIC, 8) != 0) {
        fprintf(stderr, "qmlog_read: bad magic\n");
        return false;
    }
    if (fread(&h.version,  4, 1, fp) != 1) return false;
    if (h.version != QMLOG_VERSION) {
        fprintf(stderr, "qmlog_read: unsupported version %u\n", h.version);
        return false;
    }
    if (fread(&h.n_vocab,  4, 1, fp) != 1) return false;
    if (fread(&h.n_tokens, 4, 1, fp) != 1) return false;
    if (fread(&h.n_prompt, 4, 1, fp) != 1) return false;
    if (fread(&h.temp,     4, 1, fp) != 1) return false;
    if (fread(&h.top_p,    4, 1, fp) != 1) return false;
    if (fread(&h.top_k,    4, 1, fp) != 1) return false;
    if (fread(&h.seed,     4, 1, fp) != 1) return false;

    // skip reserved
    if (fseek(fp, 32, SEEK_CUR) != 0) return false;

    return true;
}

bool qmlog_read(const std::string & path, qmlog_file & f) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "qmlog_read: cannot open '%s'\n", path.c_str());
        return false;
    }

    if (!read_header(fp, f.header)) {
        fclose(fp);
        return false;
    }

    const auto & h = f.header;

    // tokens
    f.tokens.resize(h.n_tokens);
    if ((int)fread(f.tokens.data(), 4, h.n_tokens, fp) != h.n_tokens) {
        fprintf(stderr, "qmlog_read: truncated tokens\n");
        fclose(fp);
        return false;
    }

    // logits
    const size_t n_logits = (size_t)(h.n_tokens - 1) * h.n_vocab;
    f.logits.resize(n_logits);
    if (fread(f.logits.data(), 4, n_logits, fp) != n_logits) {
        fprintf(stderr, "qmlog_read: truncated logits\n");
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}

bool qmlog_read_tokens(const std::string & path, qmlog_file & f) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "qmlog_read_tokens: cannot open '%s'\n", path.c_str());
        return false;
    }

    if (!read_header(fp, f.header)) {
        fclose(fp);
        return false;
    }

    f.tokens.resize(f.header.n_tokens);
    if ((int)fread(f.tokens.data(), 4, f.header.n_tokens, fp) != f.header.n_tokens) {
        fprintf(stderr, "qmlog_read_tokens: truncated tokens\n");
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}
