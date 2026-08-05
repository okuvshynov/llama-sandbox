#include "logits_file.h"

#include <cstdio>
#include <cstring>

namespace {

bool write_raw(FILE * fp, const void * data, size_t size) {
    return fwrite(data, 1, size, fp) == size;
}

bool write_u32(FILE * fp, uint32_t v)  { return write_raw(fp, &v, sizeof(v)); }
bool write_i32(FILE * fp, int32_t v)   { return write_raw(fp, &v, sizeof(v)); }

bool write_str(FILE * fp, const std::string & s) {
    return write_u32(fp, (uint32_t)s.size()) && write_raw(fp, s.data(), s.size());
}

bool read_raw(FILE * fp, void * data, size_t size) {
    return fread(data, 1, size, fp) == size;
}

bool read_u32(FILE * fp, uint32_t & v) { return read_raw(fp, &v, sizeof(v)); }
bool read_i32(FILE * fp, int32_t & v)  { return read_raw(fp, &v, sizeof(v)); }

bool read_str(FILE * fp, std::string & s) {
    uint32_t n = 0;
    if (!read_u32(fp, n)) return false;
    s.resize(n);
    return read_raw(fp, &s[0], n);
}

} // namespace

bool lkld_write(const std::string & path, const lkld_file & f) {
    FILE * fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "lkld_write: cannot open '%s'\n", path.c_str());
        return false;
    }

    bool ok = write_raw(fp, LKLD_MAGIC, 8)
           && write_u32(fp, LKLD_VERSION)
           && write_i32(fp, f.n_vocab)
           && write_i32(fp, f.top_k)
           && write_i32(fp, (int32_t)f.seqs.size())
           && write_str(fp, f.model_desc);

    for (const lkld_seq & s : f.seqs) {
        if (!ok) break;
        ok = write_str(fp, s.label)
          && write_i32(fp, s.n_prompt)
          && write_i32(fp, s.n_total)
          && write_i32(fp, (int32_t)s.positions.size())
          && write_raw(fp, s.tokens.data(), s.tokens.size() * sizeof(int32_t));
        for (const lkld_position & p : s.positions) {
            if (!ok) break;
            ok = write_raw(fp, &p.max_logit, sizeof(float))
              && write_raw(fp, &p.lse_rest, sizeof(float))
              && write_raw(fp, p.ids.data(),    p.ids.size()    * sizeof(int32_t))
              && write_raw(fp, p.logits.data(), p.logits.size() * sizeof(float));
        }
    }

    if (fclose(fp) != 0) ok = false;
    if (!ok) {
        fprintf(stderr, "lkld_write: write failed for '%s'\n", path.c_str());
    }
    return ok;
}

bool lkld_read(const std::string & path, lkld_file & f) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "lkld_read: cannot open '%s'\n", path.c_str());
        return false;
    }

    char magic[8];
    uint32_t version = 0;
    int32_t n_seq = 0;
    bool ok = read_raw(fp, magic, 8)
           && memcmp(magic, LKLD_MAGIC, 8) == 0
           && read_u32(fp, version)
           && version == LKLD_VERSION
           && read_i32(fp, f.n_vocab)
           && read_i32(fp, f.top_k)
           && read_i32(fp, n_seq)
           && read_str(fp, f.model_desc);

    if (ok) f.seqs.resize(n_seq);
    for (int32_t si = 0; ok && si < n_seq; si++) {
        lkld_seq & s = f.seqs[si];
        int32_t n_scored = 0;
        ok = read_str(fp, s.label)
          && read_i32(fp, s.n_prompt)
          && read_i32(fp, s.n_total)
          && read_i32(fp, n_scored);
        if (!ok) break;
        s.tokens.resize(s.n_total);
        ok = read_raw(fp, s.tokens.data(), s.tokens.size() * sizeof(int32_t));
        s.positions.resize(n_scored);
        for (lkld_position & p : s.positions) {
            if (!ok) break;
            p.ids.resize(f.top_k);
            p.logits.resize(f.top_k);
            ok = read_raw(fp, &p.max_logit, sizeof(float))
              && read_raw(fp, &p.lse_rest, sizeof(float))
              && read_raw(fp, p.ids.data(),    p.ids.size()    * sizeof(int32_t))
              && read_raw(fp, p.logits.data(), p.logits.size() * sizeof(float));
        }
    }

    if (ok && fgetc(fp) != EOF) {
        fprintf(stderr, "lkld_read: trailing bytes in '%s'\n", path.c_str());
        ok = false;
    }
    fclose(fp);
    if (!ok) {
        fprintf(stderr, "lkld_read: failed to parse '%s'\n", path.c_str());
    }
    return ok;
}
